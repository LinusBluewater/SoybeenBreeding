"""
CNN 架构调优：系统性尝试多种架构，确认能否优于 Ridge

尝试的架构方向:
  1. 超小网络 (1层卷积, 极少参数)
  2. 大卷积核 (模拟 LR 的加性效应)
  3. 深度可分离卷积 (减少参数)
  4. 残差连接
  5. 不同正则化强度
  6. 不同学习率/优化器

每种架构: 5-fold CV 选最优, 然后 test 评估
若所有架构 test r 都远低于 Ridge (0.756), 则确认 CNN 不适用
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.blup import extract_blup

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

N_FOLDS = 5


def arch_tiny(n_snps):
    """超小: 1 层卷积, 极少参数"""
    return models.Sequential([
        layers.Input(shape=(n_snps, 1)),
        layers.Conv1D(4, kernel_size=1, activation="linear"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1),
    ])


def arch_large_kernel(n_snps):
    """大卷积核: 模拟加性效应的宽窗口"""
    return models.Sequential([
        layers.Input(shape=(n_snps, 1)),
        layers.Conv1D(8, kernel_size=50, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1),
    ])


def arch_separable(n_snps):
    """深度可分离卷积 (参数少)"""
    return models.Sequential([
        layers.Input(shape=(n_snps, 1)),
        layers.SeparableConv1D(16, kernel_size=10, activation="relu", padding="same"),
        layers.MaxPooling1D(4),
        layers.SeparableConv1D(32, kernel_size=5, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])


def arch_residual(n_snps):
    """残差连接"""
    inp = layers.Input(shape=(n_snps, 1))
    x = layers.Conv1D(16, 8, padding="same", activation="relu")(inp)
    for _ in range(2):
        h = layers.Conv1D(16, 8, padding="same", activation="relu")(x)
        h = layers.BatchNormalization()(h)
        x = layers.Add()([x, h])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1)(x)
    return models.Model(inp, out)


def arch_wide_conv(n_snps):
    """宽卷积 + 大 dense"""
    return models.Sequential([
        layers.Input(shape=(n_snps, 1)),
        layers.Conv1D(32, kernel_size=20, activation="relu", padding="same"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-3)),
        layers.Dropout(0.5),
        layers.Dense(1),
    ])


ARCHS = {
    "tiny_1x1conv": arch_tiny,
    "large_kernel50": arch_large_kernel,
    "separable": arch_separable,
    "residual": arch_residual,
    "wide_conv20": arch_wide_conv,
}


def cv_cnn(make_model, X, y, kf, epochs=150):
    """5-fold CV (每折早停)"""
    rs = []
    for tr, te in kf.split(X):
        tf.keras.backend.clear_session()
        m = make_model()
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        m.fit(X[tr], y[tr], validation_data=(X[te], y[te]),
              epochs=epochs, batch_size=64, callbacks=[es], verbose=0)
        p = m.predict(X[te], verbose=0).flatten()
        rs.append(np.corrcoef(y[te], p)[0, 1])
    return np.mean(rs)


def main():
    t0 = time.time()
    print("=" * 60)
    print("CNN 架构调优 (对比 Ridge 0.756)")
    print("=" * 60)

    geno = pd.read_csv("data/geno.csv", index_col=0)
    pheno = pd.read_csv("data/pheno.csv", index_col=0)
    blup_df = extract_blup(pheno, "yield", verbose=False)["blup_df"]

    common = geno.index.intersection(blup_df.index)
    X_raw = geno.loc[common].values.astype(np.float32)
    y = blup_df.loc[common, "gebv"].values.astype(np.float32)
    strains = np.array(common)

    train_s, temp_s = train_test_split(strains, test_size=0.4, random_state=SEED)
    val_s, test_s = train_test_split(temp_s, test_size=0.5, random_state=SEED)
    idx = pd.Series(strains)
    tr_m, va_m, te_m = idx.isin(train_s).values, idx.isin(val_s).values, idx.isin(test_s).values

    imp = SimpleImputer(strategy="most_frequent")
    scl = StandardScaler()
    X_all = scl.fit_transform(imp.fit_transform(X_raw))
    X_trval, y_trval = X_all[tr_m | va_m], y[tr_m | va_m]
    X_te, y_te = X_all[te_m], y[te_m]

    X_trval = X_trval[..., None]
    X_te = X_te[..., None]
    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)

    results = []
    for name, fn in ARCHS.items():
        print(f"\n[{name}]")
        n_params = fn(X_trval.shape[1]).count_params()
        r_cv = cv_cnn(lambda: fn(X_trval.shape[1]), X_trval, y_trval, kf)
        # test
        tf.keras.backend.clear_session()
        m = fn(X_trval.shape[1])
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        m.fit(X_trval, y_trval, validation_split=0.2, epochs=200,
              batch_size=64, callbacks=[es], verbose=0)
        p = m.predict(X_te, verbose=0).flatten()
        r_te = np.corrcoef(y_te, p)[0, 1]
        print(f"  参数: {n_params:,}  CV r={r_cv:.4f}  test r={r_te:.4f}")
        results.append({"架构": name, "参数": n_params, "CV_r": r_cv, "test_r": r_te})

    df = pd.DataFrame(results).sort_values("test_r", ascending=False)
    print("\n" + "=" * 60)
    print("CNN 架构汇总 (对比 Ridge test r=0.756)")
    print("=" * 60)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    df.to_csv("outputs/cnn_tuning.csv", index=False, encoding="utf-8-sig")
    print(f"\n耗时: {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
