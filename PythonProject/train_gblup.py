"""
GBLUP (Genomic Best Linear Unbiased Prediction)
基因组选择的标准方法，作为 CNN 的基准对照。

模型: y = 1*mu + u + e
      u ~ N(0, G * sigma2_u)   <- 基因组关系矩阵作为协方差
      e ~ N(0, I * sigma2_e)

G 矩阵用 VanRaden (2008) 方法构建。
lambda = sigma2_e / sigma2_u 通过交叉验证选择。

数学等价于在 G 核上的岭回归，但通过亲缘关系矩阵"借力"
预测测试品种（这是 GBLUP 的核心优势）。
"""
import os
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

from src.blup import extract_blup


# ======================
# 加载配置
# ======================
with open("configs/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

os.makedirs(cfg["paths"]["model_out"], exist_ok=True)


# ======================
# 1. 加载原始数据
# ======================
print("=" * 50)
print("1. 加载原始数据")

geno = pd.read_csv(cfg["paths"]["geno_raw"], index_col=0)
pheno = pd.read_csv(cfg["paths"]["pheno_raw"], index_col=0)
print(f"  基因型: {geno.shape}")
print(f"  表型:   {pheno.shape}")


# ======================
# 2. 提取 BLUP 育种值 (GEBV)
# ======================
print("\n" + "=" * 50)
print("2. 提取 BLUP 育种值")

target = cfg["task"]["target"]
result = extract_blup(pheno, target, verbose=True)
blup_df = result["blup_df"]


# ======================
# 3. 基因型与育种值对齐
# ======================
print("\n" + "=" * 50)
print("3. 对齐基因型与育种值")

common = geno.index.intersection(blup_df.index)
geno_a = geno.loc[common].values.astype(np.float64)
y = blup_df.loc[common, "gebv"].values.astype(np.float64)
strains = np.array(common)
print(f"  对齐后: {geno_a.shape[0]} 品种 x {geno_a.shape[1]} SNP")
print(f"  GEBV: mean={y.mean():.1f}, std={y.std():.1f}")


# ======================
# 4. 构建基因组关系矩阵 G (VanRaden 2008)
# ======================
print("\n" + "=" * 50)
print("4. 构建 G 矩阵 (VanRaden)")

X = geno_a
# 次等位基因频率 (忽略 NA)
p = np.nanmean(X / 2.0, axis=0)
p = np.nan_to_num(p, nan=0.5)
# 中心化: X_c = X - 2p (NA 填 0)
X_c = X - 2 * p
X_c = np.nan_to_num(X_c, nan=0.0)
# VanRaden 分母: 2 * sum(p(1-p))
denom = 2 * np.sum(p * (1 - p))
G = X_c @ X_c.T / denom
# 对称化 + 对角线正则
G = (G + G.T) / 2
np.fill_diagonal(G, np.maximum(np.diag(G), 1e-6))
print(f"  G 矩阵: {G.shape}")
print(f"  对角线均值: {np.diag(G).mean():.3f} (理论~1)")
print(f"  非对角线均值: {G[~np.eye(G.shape[0], dtype=bool)].mean():.4f} (理论~0)")


# ======================
# 5. 按品种分组拆分
# ======================
print("\n" + "=" * 50)
print("5. 拆分 train/val/test")

ts, vs = cfg["data"]["test_size"], cfg["data"]["val_size"]
train_s, temp_s = train_test_split(
    strains, test_size=ts + vs, random_state=cfg["data"]["random_state"]
)
val_s, test_s = train_test_split(
    temp_s, test_size=ts / (ts + vs), random_state=cfg["data"]["random_state"]
)

idx_series = pd.Series(strains)
train_idx = np.where(idx_series.isin(train_s).values)[0]
val_idx = np.where(idx_series.isin(val_s).values)[0]
test_idx = np.where(idx_series.isin(test_s).values)[0]
print(f"  train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")


# ======================
# 6. GBLUP 核心算法 (用特征分解加速)
# ======================
print("\n" + "=" * 50)
print("6. GBLUP 拟合")

# 特征分解 G = Q D Q'  (一次性, 之后不同 lambda 都是 O(n^2))
print("  特征分解 G ...")
eigvals, eigvecs = np.linalg.eigh(G)
# 保证非负 (数值误差可能导致小负值)
eigvals = np.maximum(eigvals, 0)
print(f"  特征值范围: [{eigvals.min():.4f}, {eigvals.max():.4f}]")


def gblup_predict(y, fit_idx, pred_idx, lam, eigvals, eigvecs):
    """
    u_hat = (G + lam*I)^-1 * y_centered
          = Q (D + lam*I)^-1 Q' y_centered

    y_centered: fit 个体有值, 其他为 0
    预测 = mu + u_hat[pred_idx]
    """
    mu = y[fit_idx].mean()
    y_full = np.zeros_like(y)
    y_full[fit_idx] = y[fit_idx] - mu
    # Q' y_full
    Qt_y = eigvecs.T @ y_full
    # (D + lam)^-1
    u_hat = eigvecs @ (Qt_y / (eigvals + lam))
    return mu + u_hat[pred_idx]


# ======================
# 7. 交叉验证选 lambda (在 train 上做 5-fold)
# ======================
print("\n" + "=" * 50)
print("7. 交叉验证选择 lambda")

lambdas = [0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
best_lam, best_r2 = None, -np.inf

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for lam in lambdas:
    fold_r2 = []
    for tr, te in kf.split(train_idx):
        tr_i = train_idx[tr]
        te_i = train_idx[te]
        pred = gblup_predict(y, tr_i, te_i, lam, eigvals, eigvecs)
        if y[te_i].std() > 0:
            fold_r2.append(r2_score(y[te_i], pred))
    mean_r2 = np.mean(fold_r2)
    print(f"  lambda={lam:6.2f}  CV R2 = {mean_r2:.4f}")
    if mean_r2 > best_r2:
        best_r2, best_lam = mean_r2, lam

print(f"\n  >>> 最优 lambda = {best_lam}, CV R2 = {best_r2:.4f}")


# ======================
# 8. 最终评估 (train+val 拟合, test 评估)
# ======================
print("\n" + "=" * 50)
print("8. 最终评估")

fit_idx = np.concatenate([train_idx, val_idx])
pred_test = gblup_predict(y, fit_idx, test_idx, best_lam, eigvals, eigvecs)

r2_test = r2_score(y[test_idx], pred_test)
rmse_test = np.sqrt(mean_squared_error(y[test_idx], pred_test))
mae_test = mean_absolute_error(y[test_idx], pred_test)
corr_test = np.corrcoef(y[test_idx], pred_test)[0, 1]

print("\n" + "=" * 50)
print("========== GBLUP 结果 ==========")
print(f"目标性状:   {target} (育种值 GEBV)")
print(f"遗传力 h2:   {result['var_comp']['h2']:.3f}")
print(f"最优 lambda: {best_lam}")
print(f"R2:         {r2_test:.4f}")
print(f"RMSE:       {rmse_test:.4f}")
print(f"MAE:        {mae_test:.4f}")
print(f"预测相关 r:  {corr_test:.4f}  <- 育种界核心指标")
print("=" * 50)


# ======================
# 9. 保存育种值预测结果
# ======================
out_path = os.path.join(cfg["paths"]["blup_out"], f"gblup_pred_{target}.csv")
os.makedirs(cfg["paths"]["blup_out"], exist_ok=True)

pred_all = gblup_predict(y, fit_idx, np.arange(len(y)), best_lam, eigvals, eigvecs)
out_df = pd.DataFrame(
    {"strain": strains, "gebv_true": y, "gebv_pred": pred_all}
)
out_df.to_csv(out_path, index=False)
print(f"预测结果已保存: {out_path}")
