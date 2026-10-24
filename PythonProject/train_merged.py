"""
合并两群体训练：统一 SNP 命名 + 群体内 z-score 批次校正 + Ridge

数据:
  559群体: data/geno_physic.csv (S01_10679 命名) + kong-gwas-QD.txt
  600群体: data/geno_600.csv   (chr1_742 命名)   + 考种表(5株均值, 清洗异常值)

流程:
  1. 统一 SNP 命名 (S01_10679 -> chr1_10679), 取共同 SNP (约74.7%重叠)
  2. 表型对齐 6 个性状, 600群体考种表清洗异常值
  3. 群体内 z-score 标准化 (消除两群体系统差异/批次效应)
  4. 合并 ~825 品种, Ridge 5-fold CV
  5. 对比: 559单独 / 600单独 / 合并

回答: 合并训练能否提升预测精度?
"""
import re
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

GENO_559 = "data/geno_559_aligned.csv"   # 已按 VCF 抽样位置对齐
GENO_600 = "data/geno_600.csv"
PHENO_559 = "PhysicData/kong-gwas-QD.txt"
KAOZHONG = "PhysicData/2022潍坊农科院考种表-给计算机.xlsx"

# 性状对应: 统一名 -> (559列, 600列, 异常值上限)
TRAITS_MAP = {
    "百粒重": ("100SW", "百粒重g", 100),
    "荚数": ("JIAS", "单株荚数", None),
    "粒数": ("LS", "粒数", None),
    "株高": ("ZG", "株高cm", None),
    "节数": ("JS", "主茎节数", None),
    "单株粒重": ("DZ", "单株粒重g", 500),
}

ALPHAS = [1000, 5000, 10000, 50000]
N_FOLDS = 5
SEED = 42


def parse_snp_559(name):
    """S01_10679 -> chr1_10679; 非标准命名返回 None"""
    m = re.match(r"S(\d+)_(\d+)", name)
    if m:
        return f"chr{int(m.group(1))}_{m.group(2)}"
    return None


def load_genotypes():
    """读两个群体的基因型, 统一 SNP 命名, 取共同 SNP"""
    print("=" * 50)
    print("1. 加载基因型, 统一 SNP 命名")

    geno559 = pd.read_csv(GENO_559, index_col=0)
    geno600 = pd.read_csv(GENO_600, index_col=0)
    print(f"  559群体: {geno559.shape[0]}样本 × {geno559.shape[1]} SNP (已对齐)")
    print(f"  600群体: {geno600.shape[0]}样本 × {geno600.shape[1]} SNP")

    # 取共同 SNP (两者列名已是统一格式 chrX_pos)
    common = sorted(set(geno559.columns) & set(geno600.columns))
    print(f"  共同 SNP: {len(common)} 个 "
          f"(559的{len(common)/len(geno559.columns)*100:.0f}%, 600的{len(common)/len(geno600.columns)*100:.0f}%)")

    geno559 = geno559[common]
    geno600 = geno600[common]
    return geno559, geno600, common


def load_pheno_559():
    """559群体表型 (kong-gwas)"""
    pheno = pd.read_csv(PHENO_559, sep="\t", index_col=0)
    pheno = pheno.apply(pd.to_numeric, errors="coerce")
    return pheno


def load_pheno_600():
    """600群体表型 (考种表, 5株均值, 清洗异常值)"""
    df = pd.read_excel(KAOZHONG, sheet_name="840份大豆品种")
    df["种植编号"] = df["种植编号"].ffill()
    df["num"] = df["种植编号"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)

    result = {}
    for label, (col559, col600, clip_high) in TRAITS_MAP.items():
        sub = df[["num", col600]].rename(columns={"num": "品种", col600: "y"})
        sub["y"] = pd.to_numeric(sub["y"], errors="coerce")
        sub = sub.dropna()
        if clip_high:   # 清洗异常值 (如百粒重 19339)
            n_before = len(sub)
            sub = sub[sub["y"] <= clip_high]
            if len(sub) < n_before:
                print(f"  [清洗] {label}: 剔除 {n_before - len(sub)} 个异常值 (> {clip_high})")
        result[label] = sub.groupby("品种")["y"].mean()
    return pd.DataFrame(result)


def cv_ridge(X, y, n_folds=N_FOLDS):
    """5-fold CV, 返回最优 (r2, r, alpha)"""
    best_alpha, best_r2, best_r = None, -np.inf, 0
    for a in ALPHAS:
        kf = KFold(n_folds, shuffle=True, random_state=SEED)
        r2s, rs = [], []
        for tr, te in kf.split(X):
            m = Ridge(alpha=a); m.fit(X[tr], y[tr]); p = m.predict(X[te])
            r2s.append(r2_score(y[te], p)); rs.append(np.corrcoef(y[te], p)[0, 1])
        mr2, mr = np.mean(r2s), np.mean(rs)
        if mr2 > best_r2:
            best_r2, best_r, best_alpha = mr2, mr, a
    return best_r2, best_r, best_alpha


def main():
    geno559, geno600, common = load_genotypes()
    pheno559 = load_pheno_559()
    pheno600 = load_pheno_600()

    # 600 的样本名 AAF0001-W -> 数字; 表型600用考种表num
    # 600基因型和600表型通过 AAF数字 <-> 考种num 对应(数字相同)

    print("\n" + "=" * 50)
    print("2. 合并训练对比 (单独 vs 合并)")
    print("=" * 50)
    print(f"{'性状':10s} {'559单独':>10s} {'600单独':>10s} {'合并':>10s} {'合并-最优单独':>14s}")

    summary = []
    for label, (col559, col600, clip_high) in TRAITS_MAP.items():
        # ---- 559 单独 ----
        y559 = pheno559[col559].dropna()
        idx559 = y559.index
        X559 = geno559.loc[idx559].values.astype(np.float32)
        X559 = SimpleImputer(strategy="most_frequent").fit_transform(X559)
        X559 = StandardScaler().fit_transform(X559)
        r2_559, r_559, _ = cv_ridge(X559, y559.values)

        # ---- 600 单独 ----
        y600 = pheno600[label].dropna()
        # 600基因型样本名 AAF0001-W -> 数字, 对应考种num
        nums600 = geno600.index.str.extract(r"AAF(\d+)-", expand=False).astype(int).values
        idx600 = [i for i, n in enumerate(nums600) if n in y600.index]
        X600 = geno600.iloc[idx600].values.astype(np.float32)
        X600 = SimpleImputer(strategy="most_frequent").fit_transform(X600)
        X600 = StandardScaler().fit_transform(X600)
        y600_aligned = y600.loc[[nums600[i] for i in idx600]].values
        r2_600, r_600, _ = cv_ridge(X600, y600_aligned)

        # ---- 合并 (群体内 z-score 批次校正) ----
        # 559: z-score
        z559 = (y559 - y559.mean()) / y559.std()
        # 600: z-score
        z600 = (y600_aligned - y600_aligned.mean()) / y600_aligned.std()

        X_merged = np.vstack([X559, X600])
        y_merged = np.concatenate([z559.values, z600])
        r2_merged, r_merged, _ = cv_ridge(X_merged, y_merged)

        best_single = max(r_559, r_600)
        delta = r_merged - best_single
        print(f"{label:10s} {r_559:>10.4f} {r_600:>10.4f} {r_merged:>10.4f} {delta:>+14.4f}")
        summary.append({"性状": label, "559": r_559, "600": r_600,
                        "合并": r_merged, "delta": delta, "n合并": len(y_merged)})

    print("\n" + "=" * 50)
    print("汇总")
    print("=" * 50)
    df = pd.DataFrame(summary)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
