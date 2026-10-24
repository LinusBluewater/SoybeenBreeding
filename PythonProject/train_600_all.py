"""
600群体 全性状训练（简单均值, 因为 BLUP 已证明无效）

数据:
  基因型: data/geno_600.csv (426 AAF样本)
  表型:   考种表 2022 (5株重复, 取简单均值)

对比 559群体 的结果, 选出各性状最优
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

GENO_PATH = "data/geno_600.csv"
KAOZHONG = "PhysicData/2022潍坊农科院考种表-给计算机.xlsx"

# 性状: 考种表列名 -> 对应559群体的性状
TRAITS = {
    "株高cm": "株高(ZG)",
    "主茎节数": "节数(JS)",
    "单株荚数": "荚数(JIAS)",
    "粒数": "粒数(LS)",
    "百粒重g": "百粒重(100SW)",
    "单株粒重g": "单株粒重(DZ)",
}

ALPHAS = [1000, 5000, 10000, 50000]
N_FOLDS = 5
SEED = 42


def cv_ridge(X, y):
    best_alpha, best_r2 = None, -np.inf
    best_r = 0
    for a in ALPHAS:
        kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
        r2s, rs = [], []
        for tr, te in kf.split(X):
            m = Ridge(alpha=a); m.fit(X[tr], y[tr]); p = m.predict(X[te])
            r2s.append(r2_score(y[te], p)); rs.append(np.corrcoef(y[te], p)[0,1])
        mr2, mr = np.mean(r2s), np.mean(rs)
        if mr2 > best_r2:
            best_r2, best_r, best_alpha = mr2, mr, a
    return best_r2, best_r, best_alpha


def main():
    geno = pd.read_csv(GENO_PATH, index_col=0)
    geno_nums = geno.index.str.extract(r"AAF(\d+)-", expand=False).astype(int).values

    df = pd.read_excel(KAOZHONG, sheet_name="840份大豆品种")
    df["种植编号"] = df["种植编号"].ffill()
    df["num"] = df["种植编号"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)

    # 预处理基因型 (一次)
    X_full = SimpleImputer(strategy="most_frequent").fit_transform(geno.values.astype(np.float32))
    X_full = StandardScaler().fit_transform(X_full)

    print("=" * 60)
    print("600群体 全性状训练结果 (简单均值)")
    print("=" * 60)
    print(f"{'性状':14s} {'n':>5s} {'CV R2':>8s} {'CV r':>8s} {'最优alpha':>10s}")

    results = []
    for col, label in TRAITS.items():
        sub = df[["num", col]].rename(columns={"num": "品种", col: "y"})
        sub["y"] = pd.to_numeric(sub["y"], errors="coerce")
        sub = sub.dropna(subset=["y"])
        sub = sub[sub["品种"].isin(geno_nums)]

        mean_df = sub.groupby("品种")["y"].mean()

        # 对齐基因型和表型
        common = [n for n in geno_nums if n in mean_df.index]
        geno_idx = [i for i, n in enumerate(geno_nums) if n in common]
        X = X_full[geno_idx]
        y = mean_df.loc[common].values

        r2, r, alpha = cv_ridge(X, y)
        print(f"{label:14s} {len(y):>5d} {r2:>8.4f} {r:>8.4f} {alpha:>10d}")
        results.append({"性状": label, "n": len(y), "r": r, "r2": r2})

    print()
    return results


if __name__ == "__main__":
    main()
