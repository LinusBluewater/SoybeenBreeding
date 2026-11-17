"""
SoyNAM 多模型基准测试 v2 (精简加速版)

数据: data/geno.csv + yield BLUP
拆分: 分组 60/20/20
调参: 5-fold CV (缩减网格)
评估: 独立 test r, 增量写入 CSV (防中断丢结果)
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[注意] xgboost 未安装, 跳过", flush=True)

from src.blup import extract_blup

SEED = 42
N_FOLDS = 5
OUT_CSV = "outputs/model_benchmark.csv"


def cv_score(make_model, X, y, kf):
    rs = []
    for tr, te in kf.split(X):
        m = make_model()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        rs.append(np.corrcoef(y[te], p)[0, 1])
    return np.mean(rs)


def save_result(results):
    pd.DataFrame(results, columns=["模型", "参数", "CV_r", "test_r"]).to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig")


def record(results, X_trval, y_trval, X_te, y_te, kf, name, make_fn, grid):
    """网格搜索 + test 评估 + 增量保存"""
    t0 = time.time()
    best_r, best_cfg = -1, None
    for cfg in grid:
        r_cv = cv_score(lambda cfg=cfg: make_fn(cfg), X_trval, y_trval, kf)
        print(f"  {name} {cfg}: CV r={r_cv:.4f}", flush=True)
        if r_cv > best_r:
            best_r, best_cfg = r_cv, cfg
    m = make_fn(best_cfg)
    m.fit(X_trval, y_trval)
    r_te = np.corrcoef(y_te, m.predict(X_te))[0, 1]
    print(f"  >>> {name} 最优 {best_cfg}: CV r={best_r:.4f}, test r={r_te:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    results.append((name, str(best_cfg), best_r, r_te))
    save_result(results)


def main():
    t_all = time.time()
    print("=" * 60, flush=True)
    print("SoyNAM 多模型基准测试 v2", flush=True)
    print("=" * 60, flush=True)

    geno = pd.read_csv("data/geno.csv", index_col=0)
    pheno = pd.read_csv("data/pheno.csv", index_col=0)
    blup_df = extract_blup(pheno, "yield", verbose=False)["blup_df"]

    common = geno.index.intersection(blup_df.index)
    X_raw = geno.loc[common].values.astype(np.float32)
    y = blup_df.loc[common, "gebv"].values.astype(np.float32)
    strains = np.array(common)
    print(f"数据: {len(common)} 品种 × {X_raw.shape[1]} SNP", flush=True)

    train_s, temp_s = train_test_split(strains, test_size=0.4, random_state=SEED)
    val_s, test_s = train_test_split(temp_s, test_size=0.5, random_state=SEED)
    idx = pd.Series(strains)
    tr_m, va_m, te_m = idx.isin(train_s).values, idx.isin(val_s).values, idx.isin(test_s).values

    imp = SimpleImputer(strategy="most_frequent")
    scl = StandardScaler()
    X_all = scl.fit_transform(imp.fit_transform(X_raw))
    X_trval, y_trval = X_all[tr_m | va_m], y[tr_m | va_m]
    X_te, y_te = X_all[te_m], y[te_m]
    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    print(f"train+val: {len(y_trval)}, test: {len(y_te)}\n", flush=True)

    results = []

    print("[1] Ridge", flush=True)
    record(results, X_trval, y_trval, X_te, y_te, kf, "Ridge",
           lambda a: Ridge(alpha=a), [1000, 3000, 10000])

    print("\n[2] ElasticNet", flush=True)
    record(results, X_trval, y_trval, X_te, y_te, kf, "ElasticNet",
           lambda cfg: ElasticNet(alpha=cfg[0], l1_ratio=cfg[1], max_iter=3000),
           [(0.1, 0.5), (1, 0.5), (10, 0.5)])

    print("\n[3] KernelRidge (RBF)", flush=True)
    record(results, X_trval, y_trval, X_te, y_te, kf, "KernelRidge",
           lambda cfg: KernelRidge(alpha=cfg[0], kernel="rbf", gamma=cfg[1]),
           [(1, 1e-4), (10, 1e-4), (1, 1e-3)])

    print("\n[4] SVR (RBF)", flush=True)
    record(results, X_trval, y_trval, X_te, y_te, kf, "SVR",
           lambda c: SVR(C=c, kernel="rbf", gamma="scale", cache_size=1000),
           [0.3, 1, 3])

    print("\n[5] RandomForest", flush=True)
    record(results, X_trval, y_trval, X_te, y_te, kf, "RandomForest",
           lambda cfg: RandomForestRegressor(n_estimators=cfg[0], max_depth=cfg[1],
                                             max_features=0.1, n_jobs=-1, random_state=SEED),
           [(200, None), (500, None)])

    if HAS_XGB:
        print("\n[6] XGBoost", flush=True)
        record(results, X_trval, y_trval, X_te, y_te, kf, "XGBoost",
               lambda cfg: xgb.XGBRegressor(
                   n_estimators=cfg[0], max_depth=cfg[1], learning_rate=cfg[2],
                   subsample=0.8, colsample_bytree=0.2, reg_lambda=10,
                   n_jobs=-1, random_state=SEED, verbosity=0),
               [(300, 4, 0.05), (600, 6, 0.05), (300, 6, 0.1)])

    print("\n[7] MLP", flush=True)
    record(results, X_trval, y_trval, X_te, y_te, kf, "MLP",
           lambda cfg: MLPRegressor(hidden_layer_sizes=cfg[0], alpha=cfg[1],
                                    max_iter=400, early_stopping=True,
                                    n_iter_no_change=20, random_state=SEED),
           [((64,), 0.1), ((128, 32), 0.1), ((256, 64), 1.0)])

    print("\n" + "=" * 60, flush=True)
    print("总汇总 (按 test r 排序)", flush=True)
    print("=" * 60, flush=True)
    df = pd.DataFrame(results, columns=["模型", "参数", "CV_r", "test_r"]).sort_values("test_r", ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    save_result(results)
    print(f"\n总耗时: {(time.time()-t_all)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
