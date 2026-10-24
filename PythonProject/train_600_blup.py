"""
600群体 BLUP 对比训练：BLUP育种值 vs 简单均值

数据:
  基因型: data/geno_600.csv (426 AAF样本 × 5万 SNP)
  表型:   PhysicData 考种表 (AAK品种, 5株重复)

流程:
  1. 提取 AAF 样本对应考种品种的 5 株荚数/粒数
  2. 目标A: 简单均值 (5株平均)
  3. 目标B: BLUP育种值 (MixedLM收缩)
  4. 分别训练 Ridge, 对比 CV r

核心问题: BLUP 校正能否提升预测精度?
"""
import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

GENO_PATH = "data/geno_600.csv"
KAOZHONG_2022 = "PhysicData/2022潍坊农科院考种表-给计算机.xlsx"

TRAITS = ["单株荚数", "粒数"]   # 有真实5株重复的性状

ALPHAS = [1000, 5000, 10000, 50000]
N_FOLDS = 5
SEED = 42


def extract_blup_and_mean(df, trait):
    """从考种表提取 AAF 对应品种的 5 株性状, 返回均值 和 BLUP"""
    df["种植编号"] = df["种植编号"].ffill()
    df["num"] = df["种植编号"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)

    sub = df[["num", trait]].rename(columns={"num": "品种", trait: "y"})
    sub["y"] = pd.to_numeric(sub["y"], errors="coerce")   # 处理 '199*' 等异常值
    sub = sub.dropna(subset=["y"])
    sub = sub[sub["品种"].isin(geno_nums)]   # 只保留有基因型的品种

    # 简单均值
    mean_df = sub.groupby("品种")["y"].mean()

    # BLUP
    md = MixedLM.from_formula("y ~ 1", data=sub, groups=sub["品种"])
    mdf = md.fit(method="bfgs", disp=False)
    su = float(mdf.cov_re.iloc[0, 0]); se = float(mdf.scale)
    h2 = su / (su + se)
    blup = pd.Series({k: float(v.iloc[0]) for k, v in mdf.random_effects.items()})

    return mean_df, blup, h2, len(sub["品种"].unique())


def cv_ridge(X, y, alpha):
    """5-fold CV 返回 (R2, r)"""
    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    r2s, rs = [], []
    for tr, te in kf.split(X):
        m = Ridge(alpha=alpha)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        r2s.append(r2_score(y[te], p))
        rs.append(np.corrcoef(y[te], p)[0, 1])
    return np.mean(r2s), np.mean(rs)


def main():
    global geno_nums

    # ---- 1. 读基因型 ----
    print("=" * 50)
    print("1. 加载基因型")
    geno = pd.read_csv(GENO_PATH, index_col=0)
    # 样本名 AAF0001-W -> 数字 1
    geno_nums = geno.index.str.extract(r"AAF(\d+)-", expand=False).astype(int).values
    print(f"  基因型: {geno.shape[0]} 样本 × {geno.shape[1]} SNP")

    # ---- 2. 读考种表 ----
    print("\n2. 读取考种表")
    df = pd.read_excel(KAOZHONG_2022, sheet_name="840份大豆品种")

    # ---- 3. 逐性状对比 ----
    results = []
    for trait in TRAITS:
        print(f"\n{'=' * 50}")
        print(f"性状: {trait}")
        mean_df, blup, h2, n = extract_blup_and_mean(df, trait)
        print(f"  有基因型+有表型的品种: {n} 个, 重复力 h2={h2:.3f}")

        # 取交集
        common_nums = mean_df.index.intersection(geno_nums)
        geno_idx = [i for i, n in enumerate(geno_nums) if n in common_nums]
        X = geno.iloc[geno_idx].values.astype(np.float32)
        # 预处理
        X = SimpleImputer(strategy="most_frequent").fit_transform(X)
        X = StandardScaler().fit_transform(X)

        # 目标: 均值 vs BLUP (对齐到相同品种顺序)
        num_to_mean = mean_df.to_dict()
        num_to_blup = blup.to_dict()
        nums_order = [geno_nums[i] for i in geno_idx]
        y_mean = np.array([num_to_mean[n] for n in nums_order])
        y_blup = np.array([num_to_blup[n] for n in nums_order])

        # 选 alpha (在均值上选, 两目标用同一 alpha 公平对比)
        best_alpha, best_r2 = None, -np.inf
        for a in ALPHAS:
            r2, r = cv_ridge(X, y_mean, a)
            if r2 > best_r2:
                best_r2, best_alpha = r2, a

        r2_mean, r_mean = cv_ridge(X, y_mean, best_alpha)
        r2_blup, r_blup = cv_ridge(X, y_blup, best_alpha)

        print(f"\n  {'目标':12s} {'CV R2':>8s} {'CV r':>8s}")
        print(f"  {'简单均值':12s} {r2_mean:>8.4f} {r_mean:>8.4f}")
        print(f"  {'BLUP育种值':12s} {r2_blup:>8.4f} {r_blup:>8.4f}")
        print(f"  {'提升':12s} {r_blup - r_mean:>+8.4f} {r_blup - r_mean:>+8.4f}")

        results.append({
            "trait": trait, "n": n, "h2": h2,
            "r_mean": r_mean, "r_blup": r_blup,
            "delta": r_blup - r_mean,
        })

    # ---- 4. 汇总 ----
    print(f"\n{'=' * 50}")
    print("汇总: BLUP vs 均值")
    print("=" * 50)
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
