"""
rrBLUP (Ridge Regression BLUP) 训练脚本

基因组预测的标准方法 (Whittaker 1997, Habier 2009)。
数学上等价于 GBLUP，但在 SNP 空间求解，避免 G 矩阵在
NAM 家系数据上的数值病态问题。

用法:
  python train_ridge.py                          # 训练 config 中的 target
  python train_ridge.py --target protein          # 训练指定性状
  python train_ridge.py --all                     # 批量训练所有性状
"""
import os
import sys
import pickle
import argparse
import yaml
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.blup import extract_blup, save_blup


# ======================
# 加载配置
# ======================
with open("configs/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

for k in ["model_out", "figure_out", "blup_out", "preprocessor_out"]:
    os.makedirs(cfg["paths"][k], exist_ok=True)


def train_one_trait(target, geno, pheno, cfg, verbose=True):
    """
    训练单个性状的 rrBLUP 模型。
    返回评估结果字典。
    """
    # ---------- 2. 提取 BLUP 育种值 ----------
    result = extract_blup(pheno, target, verbose=verbose)
    blup_df = result["blup_df"]
    save_blup(result, target, cfg["paths"]["blup_out"])

    # ---------- 3. 对齐 ----------
    common = geno.index.intersection(blup_df.index)
    geno_a = geno.loc[common].copy()
    y_df = blup_df.loc[common, ["gebv"]].copy()
    assert all(geno_a.index == y_df.index)

    # ---------- 4. 拆分 ----------
    strains = np.array(common)
    ts, vs = cfg["data"]["test_size"], cfg["data"]["val_size"]
    train_s, temp_s = train_test_split(
        strains, test_size=ts + vs, random_state=cfg["data"]["random_state"]
    )
    val_s, test_s = train_test_split(
        temp_s, test_size=ts / (ts + vs), random_state=cfg["data"]["random_state"]
    )

    X_raw = geno_a.values.astype(np.float32)
    y = y_df["gebv"].values.astype(np.float32)
    idx = pd.Series(geno_a.index)
    train_mask = idx.isin(train_s).values
    val_mask = idx.isin(val_s).values
    test_mask = idx.isin(test_s).values

    X_train_raw = X_raw[train_mask]
    X_val_raw = X_raw[val_mask]
    X_test_raw = X_raw[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    # ---------- 5. 预处理 ----------
    imputer = SimpleImputer(strategy=cfg["preprocess"]["imputer"])
    X_train = imputer.fit_transform(X_train_raw)
    X_val = imputer.transform(X_val_raw)
    X_test = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    pre_path = os.path.join(
        cfg["paths"]["preprocessor_out"], f"preprocessor_{target}.pkl"
    )
    with open(pre_path, "wb") as f:
        pickle.dump({"imputer": imputer, "scaler": scaler}, f)

    # ---------- 6. CV 选 alpha ----------
    alphas = [10, 30, 100, 300, 1000, 3000, 10000]
    best_alpha, best_r2 = None, -np.inf

    if verbose:
        print(f"\n  {'alpha':>8s}  {'CV R2':>8s}  {'CV r':>8s}")
        print("  " + "-" * 30)

    for a in alphas:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_r2, fold_r = [], []
        for tr, te in kf.split(X_train):
            m = Ridge(alpha=a)
            m.fit(X_train[tr], y_train[tr])
            p = m.predict(X_train[te])
            fold_r2.append(r2_score(y_train[te], p))
            fold_r.append(np.corrcoef(y_train[te], p)[0, 1])
        mean_r2, mean_r = np.mean(fold_r2), np.mean(fold_r)
        if verbose:
            print(f"  {a:>8d}  {mean_r2:>8.4f}  {mean_r:>8.4f}")
        if mean_r2 > best_r2:
            best_r2, best_alpha = mean_r2, a

    if verbose:
        print(f"\n  >>> 最优 alpha = {best_alpha}, CV R2 = {best_r2:.4f}")

    # ---------- 7. 训练 ----------
    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])
    model = Ridge(alpha=best_alpha)
    model.fit(X_fit, y_fit)

    # ---------- 8. 评估 ----------
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    corr = np.corrcoef(y_test, pred)[0, 1]

    if verbose:
        print("\n" + "=" * 50)
        print("========== rrBLUP 结果 ==========")
        print(f"目标性状:   {target} (育种值 GEBV)")
        print(f"遗传力 h2:   {result['var_comp']['h2']:.3f}")
        print(f"最优 alpha:  {best_alpha}")
        print(f"R2:         {r2:.4f}")
        print(f"RMSE:       {rmse:.4f}")
        print(f"MAE:        {mae:.4f}")
        print(f"预测相关 r:  {corr:.4f}  <- 育种界核心指标")
        print("=" * 50)

    # ---------- 9. 保存模型 (附带 sklearn 版本，供部署时校验) ----------
    model_path = os.path.join(cfg["paths"]["model_out"], f"ridge_{target}.pkl")
    save_obj = {
        "model": model,
        "sklearn_version": sklearn.__version__,
    }
    with open(model_path, "wb") as f:
        pickle.dump(save_obj, f)
    if verbose:
        print(f"模型已保存: {model_path} (sklearn {sklearn.__version__})")

    # ---------- 10. 散点图 ----------
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, pred, alpha=0.3, s=15, c="steelblue")
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    plt.plot(lims, lims, "r--", alpha=0.7, label="y = x")
    plt.xlabel("GEBV (true)")
    plt.ylabel("GEBV (predicted)")
    plt.title(f"rrBLUP: {target} (R2={r2:.3f}, r={corr:.3f})")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(
        cfg["paths"]["figure_out"], f"ridge_{target}_scatter.png"
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"散点图已保存: {fig_path}")

    return {
        "trait": target,
        "h2": result["var_comp"]["h2"],
        "alpha": best_alpha,
        "cv_r2": best_r2,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "n_strains": len(common),
    }


def main():
    parser = argparse.ArgumentParser(description="rrBLUP 训练")
    parser.add_argument("--target", type=str, default=None,
                        help="指定训练的性状 (如 yield, protein)")
    parser.add_argument("--all", action="store_true",
                        help="批量训练 config 中 trait_cols 的所有性状")
    args = parser.parse_args()

    # ---------- 1. 加载数据 (只加载一次) ----------
    print("=" * 50)
    print("1. 加载原始数据")
    geno = pd.read_csv(cfg["paths"]["geno_raw"], index_col=0)
    pheno = pd.read_csv(cfg["paths"]["pheno_raw"], index_col=0)
    print(f"  基因型: {geno.shape[0]} 品种 x {geno.shape[1]} SNP")
    print(f"  表型:   {pheno.shape[0]} 条记录")

    # ---------- 决定训练哪些性状 ----------
    if args.all:
        targets = cfg["task"]["trait_cols"]
        print(f"\n批量训练模式: {len(targets)} 个性状")
        print(f"  {targets}")
    elif args.target:
        targets = [args.target]
    else:
        targets = [cfg["task"]["target"]]

    # ---------- 训练 ----------
    all_results = []
    for i, target in enumerate(targets):
        print("\n" + "#" * 60)
        print(f"# [{i+1}/{len(targets)}] 性状: {target}")
        print("#" * 60)
        try:
            res = train_one_trait(target, geno, pheno, cfg, verbose=True)
            all_results.append(res)
        except Exception as e:
            print(f"  [跳过] {target}: {type(e).__name__}: {e}")

    # ---------- 汇总表 ----------
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("批量训练汇总")
        print("=" * 60)
        df = pd.DataFrame(all_results)
        df = df[["trait", "h2", "alpha", "cv_r2", "r2", "rmse", "mae", "corr", "n_strains"]]
        df.columns = ["性状", "遗传力h2", "alpha", "CV_R2", "R2", "RMSE", "MAE", "r", "品种数"]
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        summary_path = os.path.join(cfg["paths"]["model_out"], "ridge_summary.csv")
        df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n汇总表已保存: {summary_path}")


if __name__ == "__main__":
    main()
