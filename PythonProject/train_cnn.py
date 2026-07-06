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
import matplotlib.pyplot as plt

from src.cnn import build_cnn


# ======================
# load config
# ======================
with open("configs/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


# ======================
# create output folders
# ======================
os.makedirs(cfg["paths"]["model_out"], exist_ok=True)
os.makedirs(cfg["paths"]["figure_out"], exist_ok=True)


# ======================
# seed
# ======================
tf.random.set_seed(cfg["project"]["seed"])
np.random.seed(cfg["project"]["seed"])


# ======================
# load data
# ======================
geno = pd.read_csv(
    cfg["paths"]["geno"],
    index_col=0
)

pheno = pd.read_csv(
    cfg["paths"]["pheno"],
    index_col=0
)

assert all(geno.index == pheno.index)


# ======================
# target
# ======================
target = cfg["task"]["target"]

pheno = pheno.drop(
    columns=cfg["task"]["drop_cols"]
)

X = geno.values.astype(np.float32)
y = pheno[target].values.astype(np.float32)


# ======================
# split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=cfg["data"]["test_size"],
    random_state=cfg["data"]["random_state"]
)


# ======================
# impute
# ======================
imputer = SimpleImputer(
    strategy=cfg["preprocess"]["imputer"]
)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# ======================
# scale
# ======================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ======================
# reshape
# ======================
X_train = X_train[..., None]
X_test = X_test[..., None]


# ======================
# model
# ======================
model = build_cnn(
    input_shape=(X_train.shape[1], 1),
    cfg=cfg
)

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()


# ======================
# early stopping
# ======================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor=cfg["early_stop"]["monitor"],
    patience=cfg["early_stop"]["patience"],
    restore_best_weights=cfg["early_stop"]["restore_best_weights"]
)


# ======================
# train
# ======================
history = model.fit(
    X_train,
    y_train,
    validation_split=cfg["data"]["val_size"],
    epochs=cfg["train"]["epochs"],
    batch_size=cfg["train"]["batch_size"],
    callbacks=[early_stop],
    verbose=1
)


# ======================
# predict
# ======================
pred = model.predict(X_test).flatten()


# ======================
# evaluate
# ======================
r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)

print("\n========== Result ==========")
print(f"Trait : {target}")
print(f"R2    : {r2:.4f}")
print(f"RMSE  : {rmse:.4f}")
print(f"MAE   : {mae:.4f}")


# ======================
# save model
# ======================
model_path = os.path.join(
    cfg["paths"]["model_out"],
    f"cnn_{target}.keras"
)

model.save(model_path)

print(f"\nModel saved to: {model_path}")


# ======================
# plot loss
# ======================
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()

figure_path = os.path.join(
    cfg["paths"]["figure_out"],
    f"cnn_{target}_loss.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Loss curve saved to: {figure_path}")