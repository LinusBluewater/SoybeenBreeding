"""
方案2 训练脚本：BLUP 育种值 + 基因型 CNN

流程:
  1. 从多环境表型提取 BLUP 育种值（去除环境主效应）
  2. 按品种对齐基因型 ↔ 育种值
  3. 按品种分组拆分 train/val/test
  4. 基因型预处理（保存 imputer/scaler 供部署用）
  5. CNN 训练
  6. 评估 + 保存模型 + 预处理器

部署时只需基因型输入，输出预测育种值（无需环境信息）。
"""
import os
import pickle
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cnn import build_cnn
from src.blup import extract_blup, save_blup


# ======================
# GPU 显存按需增长
# ======================
gpus = tf.config.list_physical_devices("GPU")
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

os.makedirs(cfg["paths"]["model_out"], exist_ok=True)
os.makedirs(cfg["paths"]["figure_out"], exist_ok=True)
os.makedirs(cfg["paths"]["blup_out"], exist_ok=True)
os.makedirs(cfg["paths"]["preprocessor_out"], exist_ok=True)

tf.random.set_seed(cfg["project"]["seed"])
np.random.seed(cfg["project"]["seed"])


# ======================
# 1. 加载原始数据
# ======================
print("=" * 50)
print("1. 加载原始数据")

geno = pd.read_csv(cfg["paths"]["geno_raw"], index_col=0)
pheno = pd.read_csv(cfg["paths"]["pheno_raw"], index_col=0)
print(f"  基因型: {geno.shape[0]} 品种 × {geno.shape[1]} SNP")
print(f"  表型:   {pheno.shape[0]} 条记录")


# ======================
# 2. 提取 BLUP 育种值
# ======================
print("\n" + "=" * 50)
print("2. 提取 BLUP 育种值")

blup_cfg = cfg["blup"]
target = cfg["task"]["target"]

result = extract_blup(
    pheno_df=pheno,
    trait=target,
    env_col=blup_cfg["env_col"],
    group_col=blup_cfg["group_col"],
    method=blup_cfg["method"],
    maxiter=blup_cfg["maxiter"],
    min_envs_per_strain=blup_cfg["min_envs_per_strain"],
    verbose=True,
)

blup_df = result["blup_df"]
blup_path = save_blup(result, target, cfg["paths"]["blup_out"])
print(f"  BLUP 已保存: {blup_path}")


# ======================
# 3. 基因型与育种值按品种对齐
# ======================
print("\n" + "=" * 50)
print("3. 基因型与育种值对齐")

# 取交集
common = geno.index.intersection(blup_df.index)
print(f"  基因型品种: {len(geno.index)}")
print(f"  BLUP品种:   {len(blup_df)}")
print(f"  交集:       {len(common)}")

geno_aligned = geno.loc[common].copy()
y_df = blup_df.loc[common, ["gebv"]].copy()  # 用 GEBV（有实际单位）

# 确保对齐
assert all(geno_aligned.index == y_df.index), "品种未对齐"
print(f"  对齐后: {geno_aligned.shape[0]} 品种 × {geno_aligned.shape[1]} SNP")


# ======================
# 4. 按品种分组拆分
# ======================
print("\n" + "=" * 50)
print("4. 按品种分组拆分 train/val/test")

strains = np.array(common)
ts, vs = cfg["data"]["test_size"], cfg["data"]["val_size"]

train_s, temp_s = train_test_split(
    strains, test_size=ts + vs, random_state=cfg["data"]["random_state"]
)
val_s, test_s = train_test_split(
    temp_s,
    test_size=ts / (ts + vs),
    random_state=cfg["data"]["random_state"],
)

X = geno_aligned.values.astype(np.float32)
y = y_df["gebv"].values.astype(np.float32)
idx = pd.Series(geno_aligned.index)

train_mask = idx.isin(train_s).values
val_mask = idx.isin(val_s).values
test_mask = idx.isin(test_s).values

X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

print(f"  训练: {len(y_train)} 品种")
print(f"  验证: {len(y_val)} 品种")
print(f"  测试: {len(y_test)} 品种")
print(f"  泄露检查 (应全为0): "
      f"{len(set(train_s) & set(val_s))}, "
      f"{len(set(train_s) & set(test_s))}, "
      f"{len(set(val_s) & set(test_s))}")


# ======================
# 5. 基因型预处理（保存供部署用）
# ======================
print("\n" + "=" * 50)
print("5. 基因型预处理")

imputer = SimpleImputer(strategy=cfg["preprocess"]["imputer"])
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 保存预处理器（部署时必须用同一个 imputer/scaler）
preprocessor_path = os.path.join(
    cfg["paths"]["preprocessor_out"], f"preprocessor_{target}.pkl"
)
with open(preprocessor_path, "wb") as f:
    pickle.dump({"imputer": imputer, "scaler": scaler}, f)
print(f"  预处理器已保存: {preprocessor_path}")

# reshape for Conv1D
X_train = X_train[..., None]
X_val = X_val[..., None]
X_test = X_test[..., None]
print(f"  输入形状: {X_train.shape}")


# ======================
# 6. 构建模型
# ======================
print("\n" + "=" * 50)
print("6. 构建模型")

model = build_cnn(
    input_shape=(X_train.shape[1], 1),
    cfg=cfg,
)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()


# ======================
# 7. 训练
# ======================
print("\n" + "=" * 50)
print("7. 开始训练")

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor=cfg["early_stop"]["monitor"],
    patience=cfg["early_stop"]["patience"],
    restore_best_weights=cfg["early_stop"]["restore_best_weights"],
)

checkpoint_path = os.path.join(
    cfg["paths"]["model_out"], f"cnn_blup_{target}_ckpt.h5"
)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor=cfg["early_stop"]["monitor"],
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=cfg["train"]["epochs"],
    batch_size=cfg["train"]["batch_size"],
    callbacks=[early_stop, checkpoint],
    verbose=1,
)


# ======================
# 8. 评估
# ======================
print("\n" + "=" * 50)
print("8. 评估模型")

pred = model.predict(X_test).flatten()
r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)

# 育种值相关性（育种界更关心这个）
corr = np.corrcoef(y_test, pred)[0, 1]

print("\n" + "=" * 50)
print("========== 方案2 结果 ==========")
print(f"目标性状:   {target} (育种值 GEBV)")
print(f"遗传力 h²:   {result['var_comp']['h2']:.3f}")
print(f"R²:         {r2:.4f}")
print(f"RMSE:       {rmse:.4f}")
print(f"MAE:        {mae:.4f}")
print(f"预测相关 r:  {corr:.4f}  ← 育种界核心指标")
print("=" * 50)


# ======================
# 9. 保存模型
# ======================
model_path = os.path.join(cfg["paths"]["model_out"], f"cnn_blup_{target}.keras")
model.save(model_path)
print(f"\n模型已保存: {model_path}")


# ======================
# 10. 绘制训练曲线
# ======================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title(f"BLUP-CNN training curve ({target})")
plt.legend()
plt.tight_layout()

fig_path = os.path.join(
    cfg["paths"]["figure_out"], f"cnn_blup_{target}_loss.png"
)
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"训练曲线已保存: {fig_path}")
