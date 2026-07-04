import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
import matplotlib.pyplot as plt

from src.cnn import build_cnn


# ======================
# load config
# ======================
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)


# ======================
# seed
# ======================
tf.random.set_seed(cfg["project"]["seed"])
np.random.seed(cfg["project"]["seed"])


# ======================
# load data
# ======================
geno = pd.read_csv(cfg["paths"]["geno"], index_col=0)
pheno = pd.read_csv(cfg["paths"]["pheno"], index_col=0)

assert all(geno.index == pheno.index)


# ======================
# target
# ======================
target = cfg["task"]["target"]
pheno = pheno.drop(columns=cfg["task"]["drop_cols"])

X = geno.values.astype(np.float32)
y = pheno[target].values.astype(np.float32)


# ======================
# split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=cfg["data"]["test_size"],
    random_state=cfg["data"]["random_state"]
)


# ======================
# impute
# ======================
imp = SimpleImputer(strategy=cfg["preprocess"]["imputer"])
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)


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
model = build_cnn((X_train.shape[1], 1), cfg)

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# ======================
# early stop
# ======================
early = tf.keras.callbacks.EarlyStopping(
    monitor=cfg["early_stop"]["monitor"],
    patience=cfg["early_stop"]["patience"],
    restore_best_weights=cfg["early_stop"]["restore_best_weights"]
)


# ======================
# train
# ======================
history = model.fit(
    X_train, y_train,
    validation_split=cfg["data"]["val_size"],
    epochs=cfg["train"]["epochs"],
    batch_size=cfg["train"]["batch_size"],
    callbacks=[early],
    verbose=1
)


# ======================
# eval
# ======================
pred = model.predict(X_test).flatten()

print("R2:", r2_score(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("MAE:", mean_absolute_error(y_test, pred))


# ======================
# save
# ======================
model.save(f"{cfg['paths']['model_out']}/cnn_{target}.keras")


# ======================
# plot
# ======================
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend()
plt.show()