"""
PhysicData 阶段一训练脚本：rrBLUP 预测产量相关性状

数据:
  基因型: data/geno_physic.csv (品种 × SNP, 由 preprocess_geno.py 生成)
  表型:   PhysicData/kong-gwas-QD.txt (418 品种 × 7 性状)

性状 (7 个, 全是产量相关):
  ZG    株高
  JS    节数(主茎节数)
  JIAS  荚数
  100SW 百粒重
  DZ    单株粒重 (≈产量, 核心目标)
  JLZ   (待确认)
  LS    粒数

方法:
  样本少(418), 无多环境 -> 跳过 BLUP, 直接用 Ridge + 5-fold CV
"""
import os
import pickle
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================
# 配置
# ======================
GENO_PATH = "data/geno_physic.csv"
PHENO_PATH = "PhysicData/kong-gwas-QD.txt"

# 目标性状 (kong-gwas-QD.txt 的全部 7 个性状)
TARGETS = ["ZG", "JS", "JIAS", "100SW", "DZ", "JLZ", "LS"]

# Ridge 正则化强度搜索范围
ALPHAS = [1, 10, 100, 1000, 5000, 10000, 50000]

N_FOLDS = 5
SEED = 42

OUT_MODEL = "outputs/models/ridge_physic_{trait}.pkl"
OUT_PREP = "outputs/preprocessors/preprocessor_physic_{trait}.pkl"
OUT_FIG = "outputs/figures/ridge_physic_{trait}_scatter.png"


def load_data():
    """加载基因型和表型, 对齐样本"""
    print("=" * 50)
    print("1. 加载数据")

    geno = pd.read_csv(GENO_PATH, index_col=0)
    pheno = pd.read_csv(PHENO_PATH, sep="\t", index_col=0)

    # 强制转数值: 表型里的 Excel 错误值(如 #DIV/0!) 转成 NaN
    pheno = pheno.apply(pd.to_numeric, errors="coerce")

    print(f"  基因型: {geno.shape[0]} 品种 × {geno.shape[1]} SNP")
    print(f"  表型:   {pheno.shape[0]} 品种 × {pheno.shape[1]} 性状")

    # 对齐: 取交集
    common = geno.index.intersection(pheno.index)
    print(f"  交集品种: {len(common)}")

    geno_a = geno.loc[common]
    pheno_a = pheno.loc[common]
    return geno_a, pheno_a


def train_one_trait(X, y, trait):
    """训练单个性状的 Ridge 模型, 用 5-fold CV 评估"""
    print(f"\n{'=' * 50}")
    print(f"性状: {trait}")

    # 预处理 (填充 + 标准化)
    imputer = SimpleImputer(strategy="most_frequent")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 5-fold CV 选 alpha
    print(f"\n  {'alpha':>8s}  {'CV R2':>8s}  {'CV r':>8s}")
    print("  " + "-" * 30)

    best_alpha, best_r2 = None, -np.inf
    for a in ALPHAS:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_r2, fold_r = [], []
        for tr, te in kf.split(X):
            m = Ridge(alpha=a)
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
            fold_r2.append(r2_score(y[te], p))
            fold_r.append(np.corrcoef(y[te], p)[0, 1])
        mean_r2, mean_r = np.mean(fold_r2), np.mean(fold_r)
        print(f"  {a:>8d}  {mean_r2:>8.4f}  {mean_r:>8.4f}")
        if mean_r2 > best_r2:
            best_r2, best_alpha = mean_r2, a

    print(f"\n  最优 alpha = {best_alpha}, CV R2 = {best_r2:.4f}")

    # 用最优 alpha 训练最终模型 (全量)
    model = Ridge(alpha=best_alpha)
    model.fit(X, y)

    # 预测 (用同一份数据做 in-sample 预测看拟合程度, 真实泛化看 CV)
    pred = model.predict(X)
    r2 = r2_score(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    corr = np.corrcoef(y, pred)[0, 1]

    print(f"\n  ========== {trait} 结果 ==========")
    print(f"  样本数:     {len(y)}")
    print(f"  性状均值:   {y.mean():.2f} ± {y.std():.2f}")
    print(f"  CV R2:      {best_r2:.4f}  <- 泛化能力(关键指标)")
    print(f"  CV r:       {best_r2 and np.sqrt(max(best_r2,0)):.4f} (近似)")
    print(f"  In-sample R2: {r2:.4f} (拟合程度, 会虚高)")
    print(f"  RMSE:       {rmse:.4f}")
    print(f"  MAE:        {mae:.4f}")
    print(f"  相关 r:     {corr:.4f}")

    return {
        "trait": trait,
        "n": len(y),
        "alpha": best_alpha,
        "cv_r2": best_r2,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
    }


def main():
    geno, pheno = load_data()

    results = []
    for trait in TARGETS:
        if trait not in pheno.columns:
            print(f"\n[跳过] 性状 {trait} 不在表型中")
            continue

        # 提取目标性状, 删除缺失
        y_df = pheno[[trait]].dropna()
        common_idx = y_df.index
        X = geno.loc[common_idx].values.astype(np.float32)
        y = y_df[trait].values.astype(np.float32)

        print(f"\n{trait}: {len(y)} 个有效样本 (原始 {pheno.shape[0]})")

        res = train_one_trait(X, y, trait)
        results.append(res)

    # 汇总
    print(f"\n{'=' * 50}")
    print("汇总")
    print("=" * 50)
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
