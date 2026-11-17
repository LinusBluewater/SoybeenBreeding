"""
SoyNAM 数据训练卷积神经网络 (CNN) 预测表型

数据:
  基因型: data/geno.csv (5590品种 × 4611 SNP)
  目标:   yield 的 BLUP 育种值 (环境校正, 与 rrBLUP 一致, 公平对比)

设计要点 (吸取之前 CNN 失败的教训):
  1. 小型网络: 参数 ~5000, 远小于训练样本 3326 (防过拟合)
  2. 目标用 BLUP 育种值 (而非原始表型, 去除环境噪声)
  3. 按品种分组拆分 (防数据泄露)
  4. Dropout + Early Stopping 正则化

对比基准: Ridge (rrBLUP) 在同样拆分下 yield 测试 r=0.756
"""
import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, GlobalAveragePooling1D,
                                     Dense, Dropout, Input)

from src.blup import extract_blup

# 环境变量 (本地 Windows TF DLL 兼容)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_small_cnn(n_snps):
    """小型 1D-CNN (参数~5000, 适合小样本)"""
    model = Sequential([
        Input(shape=(n_snps, 1)),
        Conv1D(8, kernel_size=10, activation="relu", padding="same"),
        MaxPooling1D(2),
        Conv1D(16, kernel_size=10, activation="relu", padding="same"),
        MaxPooling1D(2),
        Conv1D(32, kernel_size=5, activation="relu", padding="same"),
        GlobalAveragePooling1D(),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1),
    ])
    return model


def main():
    print("=" * 50)
    print("1. 提取 BLUP 育种值 (yield)")
    geno = pd.read_csv("data/geno.csv", index_col=0)
    pheno = pd.read_csv("data/pheno.csv", index_col=0)
    result = extract_blup(pheno, "yield", verbose=True)
    blup_df = result["blup_df"]

    # 对齐
    common = geno.index.intersection(blup_df.index)
    X = geno.loc[common].values.astype(np.float32)
    y = blup_df.loc[common, "gebv"].values.astype(np.float32)
    strains = np.array(common)
    print(f"  对齐: {len(common)} 品种 × {X.shape[1]} SNP")

    # ---- 分组拆分 (与 Ridge 一致: 60/20/20) ----
    print("\n2. 按品种分组拆分")
    train_s, temp_s = train_test_split(strains, test_size=0.4, random_state=SEED)
    val_s, test_s = train_test_split(temp_s, test_size=0.5, random_state=SEED)
    idx = pd.Series(strains)
    tr_m = idx.isin(train_s).values
    va_m = idx.isin(val_s).values
    te_m = idx.isin(test_s).values
    print(f"  train: {tr_m.sum()}, val: {va_m.sum()}, test: {te_m.sum()}")

    # ---- 预处理 ----
    print("\n3. 预处理")
    imputer = SimpleImputer(strategy="most_frequent")
    X_tr = imputer.fit_transform(X[tr_m]); X_va = imputer.transform(X[va_m]); X_te = imputer.transform(X[te_m])
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr); X_va = scaler.transform(X_va); X_te = scaler.transform(X_te)
    X_tr = X_tr[..., None]; X_va = X_va[..., None]; X_te = X_te[..., None]
    print(f"  输入形状: {X_tr.shape}")

    # ---- CNN ----
    print("\n4. 训练小型 CNN")
    model = build_small_cnn(X_tr.shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    model.summary()

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True)

    history = model.fit(
        X_tr, y[tr_m],
        validation_data=(X_va, y[va_m]),
        epochs=300, batch_size=32, callbacks=[early], verbose=1)

    # ---- 评估 ----
    print("\n5. 评估 (独立测试集)")
    pred = model.predict(X_te).flatten()
    r = np.corrcoef(y[te_m], pred)[0, 1]
    r2 = r2_score(y[te_m], pred)
    rmse = np.sqrt(mean_squared_error(y[te_m], pred))

    print("\n" + "=" * 50)
    print("CNN 结果 (SoyNAM yield)")
    print("=" * 50)
    print(f"  测试 r:     {r:.4f}")
    print(f"  测试 R²:    {r2:.4f}")
    print(f"  测试 RMSE:  {rmse:.2f}")
    print(f"  对比基准:   Ridge (rrBLUP) r=0.756")
    print(f"  CNN vs Ridge: {r - 0.756:+.4f}")


if __name__ == "__main__":
    main()
