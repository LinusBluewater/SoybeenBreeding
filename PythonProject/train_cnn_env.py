import os
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # 无显示环境（SSH/服务器）后端
import matplotlib.pyplot as plt

from src.cnn import build_cnn_with_env


# ======================
# GPU 内存按需增长（避免占满显存）
# ======================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"已检测到 {len(gpus)} 个 GPU，启用显存按需增长")
    except RuntimeError as e:
        print(f"GPU 内存设置警告: {e}")
else:
    print("未检测到 GPU，使用 CPU 训练")


# ======================
# 加载配置
# ======================
with open("configs/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


# ======================
# 创建输出目录
# ======================
os.makedirs(cfg["paths"]["model_out"], exist_ok=True)
os.makedirs(cfg["paths"]["figure_out"], exist_ok=True)


# ======================
# 随机种子
# ======================
tf.random.set_seed(cfg["project"]["seed"])
np.random.seed(cfg["project"]["seed"])


# ======================
# 1. 加载原始数据
# ======================
print("=" * 50)
print("加载原始数据...")

geno = pd.read_csv(
    cfg["paths"]["geno_raw"],
    index_col=0
)

pheno = pd.read_csv(
    cfg["paths"]["pheno_raw"],
    index_col=0
)

print(f"  基因型: {geno.shape[0]} 品种 × {geno.shape[1]} SNP")
print(f"  表型:   {pheno.shape[0]} 条记录 × {pheno.shape[1]} 列")


# ======================
# 2. 提取需要的列
# ======================
target = cfg["task"]["target"]
trait_cols = cfg["task"]["trait_cols"]
env_col = cfg["task"]["env_col"]

# 只保留 strain + environ + 性状列
keep_cols = ["strain", env_col] + trait_cols
pheno = pheno[keep_cols].copy()

# 删除目标性状缺失的行
before = len(pheno)
pheno = pheno.dropna(subset=[target])
print(f"\n  删除 {target} 缺失后: {len(pheno)} 条 (丢弃 {before - len(pheno)} 条)")

print(f"  环境数: {pheno[env_col].nunique()}")
print(f"  品种数: {pheno['strain'].nunique()}")


# ======================
# 3. 环境 one-hot 编码
# ======================
print("\n" + "=" * 50)
print("环境 one-hot 编码...")

env_dummies = pd.get_dummies(pheno[env_col], prefix="env")
env_dim = env_dummies.shape[1]
print(f"  环境特征维度: {env_dim}")

# 将 one-hot 编码拼接回 pheno（后续按行索引对齐）
pheno_encoded = pd.concat(
    [pheno[["strain"] + trait_cols], env_dummies],
    axis=1
)


# ======================
# 4. 取基因型与表型的品种交集
# ======================
print("\n" + "=" * 50)
print("品种交集过滤...")

# 只保留同时存在于 geno 和 pheno 的品种
common_strains = geno.index.intersection(pheno_encoded["strain"].unique())
print(f"  基因型品种: {len(geno.index)}")
print(f"  表型品种:   {pheno_encoded['strain'].nunique()}")
print(f"  交集品种:   {len(common_strains)}")

pheno_encoded = pheno_encoded[pheno_encoded["strain"].isin(common_strains)].copy()
print(f"  过滤后表型记录: {len(pheno_encoded)}")


# ======================
# 5. 基因型按品种扩展 + 与环境特征拼接
# ======================
print("\n" + "=" * 50)
print("构建特征矩阵...")

# 每个品种的基因型向量映射到 pheno 每一行
strains = pheno_encoded["strain"].values
geno_expanded = geno.loc[strains].values.astype(np.float32)

# 环境 one-hot 特征（从过滤后的 pheno_encoded 中提取）
env_cols = [c for c in pheno_encoded.columns if c.startswith("env_")]
env_features = pheno_encoded[env_cols].values.astype(np.float32)

# 目标值
y = pheno_encoded[target].values.astype(np.float32)

print(f"  基因型特征: {geno_expanded.shape}")
print(f"  环境特征:   {env_features.shape}")
print(f"  目标 y:     {y.shape}")


# ======================
# 6. 按品种分组拆分（防止数据泄露）
# ======================
print("\n" + "=" * 50)
print("按品种分组拆分训练/验证/测试集...")

unique_strains = pheno_encoded["strain"].unique()
print(f"  独立品种数: {len(unique_strains)}")

# 第一层拆分：训练 vs (验证+测试)
train_strains, temp_strains = train_test_split(
    unique_strains,
    test_size=cfg["data"]["test_size"] + cfg["data"]["val_size"],
    random_state=cfg["data"]["random_state"]
)

# 第二层拆分：验证 vs 测试
val_strains, test_strains = train_test_split(
    temp_strains,
    test_size=cfg["data"]["test_size"] / (cfg["data"]["test_size"] + cfg["data"]["val_size"]),
    random_state=cfg["data"]["random_state"]
)

# 按品种划分行索引 —— 三者品种完全不重叠
train_mask = pheno_encoded["strain"].isin(train_strains).values
val_mask = pheno_encoded["strain"].isin(val_strains).values
test_mask = pheno_encoded["strain"].isin(test_strains).values

X_geno_train, X_geno_val, X_geno_test = (
    geno_expanded[train_mask], geno_expanded[val_mask], geno_expanded[test_mask]
)
X_env_train, X_env_val, X_env_test = (
    env_features[train_mask], env_features[val_mask], env_features[test_mask]
)
y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

print(f"  训练集: {len(y_train)} 条 (品种 {len(train_strains)})")
print(f"  验证集: {len(y_val)} 条 (品种 {len(val_strains)})")
print(f"  测试集: {len(y_test)} 条 (品种 {len(test_strains)})")
print(f"  品种不重叠检查: {len(set(train_strains) & set(val_strains))}, "
      f"{len(set(train_strains) & set(test_strains))}, "
      f"{len(set(val_strains) & set(test_strains))} (全为0则无泄露)")


# ======================
# 7. 基因型特征预处理（填充 + 标准化）
# ======================
print("\n" + "=" * 50)
print("基因型特征预处理...")

# 填充缺失 SNP
imputer = SimpleImputer(strategy=cfg["preprocess"]["imputer"])
X_geno_train = imputer.fit_transform(X_geno_train)
X_geno_val = imputer.transform(X_geno_val)
X_geno_test = imputer.transform(X_geno_test)

# 标准化
scaler = StandardScaler()
X_geno_train = scaler.fit_transform(X_geno_train)
X_geno_val = scaler.transform(X_geno_val)
X_geno_test = scaler.transform(X_geno_test)

# Reshape 为 Conv1D 输入: (samples, n_snps, 1)
X_geno_train = X_geno_train[..., None]
X_geno_val = X_geno_val[..., None]
X_geno_test = X_geno_test[..., None]

print(f"  基因型输入形状: {X_geno_train.shape}")


# ======================
# 8. 构建双输入模型
# ======================
print("\n" + "=" * 50)
print("构建模型...")

model = build_cnn_with_env(
    geno_shape=(X_geno_train.shape[1], 1),
    env_dim=env_dim,
    cfg=cfg
)

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()


# ======================
# 9. 训练
# ======================
print("\n" + "=" * 50)
print("开始训练...")

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor=cfg["early_stop"]["monitor"],
    patience=cfg["early_stop"]["patience"],
    restore_best_weights=cfg["early_stop"]["restore_best_weights"]
)

# 保存最佳模型权重，防止 SSH 断开导致训练成果丢失
checkpoint_path = os.path.join(
    cfg["paths"]["model_out"],
    f"cnn_env_{target}_ckpt.h5"
)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor=cfg["early_stop"]["monitor"],
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

history = model.fit(
    [X_geno_train, X_env_train],
    y_train,
    validation_data=([X_geno_val, X_env_val], y_val),
    epochs=cfg["train"]["epochs"],
    batch_size=cfg["train"]["batch_size"],
    callbacks=[early_stop, checkpoint],
    verbose=1
)


# ======================
# 10. 预测与评估
# ======================
print("\n" + "=" * 50)
print("评估模型...")

pred = model.predict([X_geno_test, X_env_test]).flatten()

r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)

print("\n" + "=" * 50)
print("========== 方案A 结果 ==========")
print(f"目标性状 : {target}")
print(f"环境特征维度: {env_dim}")
print(f"R²    : {r2:.4f}")
print(f"RMSE  : {rmse:.4f}")
print(f"MAE   : {mae:.4f}")
print("=" * 50)


# ======================
# 11. 保存模型
# ======================
model_path = os.path.join(
    cfg["paths"]["model_out"],
    f"cnn_env_{target}.keras"
)
model.save(model_path)
print(f"\n模型已保存: {model_path}")


# ======================
# 12. 绘制训练曲线
# ======================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()

figure_path = os.path.join(
    cfg["paths"]["figure_out"],
    f"cnn_env_{target}_loss.png"
)
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"训练曲线已保存: {figure_path}")
