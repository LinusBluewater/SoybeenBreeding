"""
GWAS 分析：600群体两年均值表型, 单SNP回归 + FDR校正

方法:
  每个SNP: y ~ SNP 线性回归, 算 p 值
  多重检验: FDR (BH) 校正
  输出: 每性状的显著SNP + 与 Ridge β 关键位点的交叉验证

局限(诚实说明):
  1. 样本量 426, 统计功效只能检测 R²>3~4% 的位点
  2. 单SNP回归未控制群体结构 (MLM 更严谨, 但慢)
  3. 显著SNP是"关联位点", LD 使邻近位点一起显著, 非严格因果
"""
import re
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from statsmodels.stats.multitest import multipletests
from scipy import stats

GENO = "data/geno_600.csv"
KAOZHONG_2022 = "PhysicData/2022潍坊农科院考种表-给计算机.xlsx"
KAOZHONG_2024 = "PhysicData/2024年潍坊农科院考种-给计算机.xlsx"

TRAITS_MAP = {
    "荚数": ("单株荚数", "单株荚数", None),
    "粒数": ("粒数", "粒数", None),
    "百粒重": ("百粒重g", "百粒重", 100),
    "单株粒重": ("单株粒重g", "单株粒重", 500),
    "株高": ("株高cm", "株高", None),
    "节数": ("主茎节数", "主茎节数", None),
}


def load_year_means(fname, sheet, header, id_col, col_map):
    df = pd.read_excel(fname, sheet_name=sheet, header=header)
    nums = df[id_col].ffill().astype(str).str.extract(r"(\d+)", expand=False)
    result = {}
    for trait, col in col_map.items():
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = np.percentile(v.dropna(), [25, 75]); iqr = q3 - q1
        mask = (v >= q1 - 3 * iqr) & (v <= q3 + 3 * iqr)
        sub = pd.DataFrame({"num": nums, "y": v}).loc[mask.values].dropna()
        sub["num"] = sub["num"].astype(int)
        result[trait] = sub.groupby("num")["y"].mean()
    return pd.DataFrame(result)


def load_pheno_two_year():
    y22 = load_year_means(KAOZHONG_2022, "840份大豆品种", 0, "种植编号",
                          {t: m[0] for t, m in TRAITS_MAP.items()})
    y24 = load_year_means(KAOZHONG_2024, "山大栽培", 1, "种植编号",
                          {t: m[1] for t, m in TRAITS_MAP.items()})
    common = sorted(set(y22.index) & set(y24.index))
    pheno = pd.DataFrame({t: (y22.loc[common, t] + y24.loc[common, t]) / 2
                          for t in TRAITS_MAP
                          if t in y22.columns and t in y24.columns})
    return pheno


def run_gwas(geno_df, pheno_df, trait, fdr_threshold=0.1):
    """单SNP回归 + FDR"""
    # 对齐 (注意: AAF-W/-X 数字可能重复, 每数字只取一个基因型样本)
    y_df = pheno_df[trait].dropna()
    sample_idx = {}
    for i, n in enumerate(geno_nums):
        if n not in sample_idx:
            sample_idx[n] = i
    aligned = [(n, sample_idx[n]) for n in y_df.index if n in sample_idx]
    nums = [a[0] for a in aligned]
    idx = [a[1] for a in aligned]
    X = geno_df.iloc[idx].values.astype(np.float64)
    y = y_df.loc[nums].values.astype(np.float64)
    n = len(y)

    # 填充 + 标准化
    col_mean = np.nanmean(X, axis=0)
    X_filled = np.where(np.isnan(X), col_mean[None, :], X)
    X_std = (X_filled - X_filled.mean(axis=0)) / (X_filled.std(axis=0) + 1e-10)
    y_std = (y - y.mean()) / (y.std() + 1e-10)

    # PCA 协变量 (前5个PC, 吸收群体结构)
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    pcs = U[:, :5]
    # 残差化: y 和每个 SNP 都去掉 PC 的影响
    P = pcs @ pcs.T
    y_resid = y_std - P @ y_std
    X_resid = X_std - P @ X_std

    # 所有SNP的相关系数 (向量化, 用残差)
    r = (X_resid.T @ y_resid) / n
    # t统计量 -> p值
    t_stat = r * np.sqrt((n - 2) / (1 - r**2 + 1e-10))
    pvals = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))

    # FDR校正
    _, fdr_q, _, _ = multipletests(pvals, method="fdr_bh")

    result = pd.DataFrame({
        "SNP": geno_df.columns,
        "chrom": [int(re.match(r"chr(\d+)_", s).group(1)) for s in geno_df.columns],
        "pos": [int(re.match(r"chr\d+_(\d+)", s).group(1)) for s in geno_df.columns],
        "r": r,
        "p": pvals,
        "FDR": fdr_q,
    })
    return result, n


def main():
    global geno_nums
    print("=" * 50)
    print("GWAS 分析 (单SNP回归 + FDR)")
    print("=" * 50)

    geno = pd.read_csv(GENO, index_col=0)
    geno_nums = geno.index.str.extract(r"AAF(\d+)-", expand=False).astype(int).values
    pheno = load_pheno_two_year()
    print(f"基因型: {geno.shape}, 表型: {pheno.shape}")

    print("\n" + "=" * 50)
    print("各性状 GWAS 结果 (FDR < 0.1)")
    print("=" * 50)

    for trait in TRAITS_MAP:
        res, n = run_gwas(geno, pheno, trait)
        sig = res[res["FDR"] < 0.1]
        n_sig_005 = (res["FDR"] < 0.05).sum()
        top_r2 = (res["r"] ** 2).max()
        print(f"\n{trait} (n={n}):")
        print(f"  FDR<0.05: {n_sig_005} 个, FDR<0.1: {len(sig)} 个, "
              f"最强位点 R²={top_r2*100:.2f}%")
        if len(sig) > 0:
            top = sig.nsmallest(5, "p")
            print(f"  Top5 显著位点:")
            for _, row in top.iterrows():
                print(f"    chr{row['chrom']}:{row['pos']}  "
                      f"p={row['p']:.2e}  FDR={row['FDR']:.3f}  "
                      f"r={row['r']:+.3f}")
        else:
            print(f"  (无 FDR<0.1 的显著位点 — 多基因性状的典型表现)")


if __name__ == "__main__":
    main()
