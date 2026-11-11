"""
育种方案设计：从候选品种中推荐杂交亲本组合

五个步骤:
  1. 综合育种值 (选择指数, 权重可调)
  2. SNP 互补性 (用 Ridge β 系数)
  3. 遗传距离 (基因组关系矩阵 G, 防近交)
  4. 多目标打分, 推荐杂交组合 Top-N
  5. 模拟杂交后代 (孟德尔分离, 预期表现)

用法:
  python select_parents.py                          # 默认: 产量+品质平衡
  python select_parents.py --weights 百粒重=0.3 单株粒重=0.3 荚数=0.2 粒数=0.2
  python select_parents.py --top 20                 # 推荐20个组合
"""
import os
import sys
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from src.blup import extract_blup  # 仅复用函数, 不调用

MODEL_DIR = "outputs/models"
PREP_DIR = "outputs/preprocessors"
GENO_600 = "data/geno_600.csv"
KAOZHONG_2022 = "PhysicData/2022潍坊农科院考种表-给计算机.xlsx"
KAOZHONG_2024 = "PhysicData/2024年潍坊农科院考种-给计算机.xlsx"

TRAITS = ["百粒重", "荚数", "粒数", "单株粒重", "株高", "节数"]

# 默认: 产量+品质平衡
DEFAULT_WEIGHTS = {"百粒重": 0.3, "单株粒重": 0.3, "荚数": 0.2, "粒数": 0.2}


def load_genotypes():
    print("=" * 50)
    print("1. 加载候选品种基因型")
    geno = pd.read_csv(GENO_600, index_col=0)
    # 426 个 AAF 样本, 每个是一个候选亲本
    print(f"  候选亲本: {geno.shape[0]} 个 × {geno.shape[1]} SNP")
    return geno


def predict_all_traits(geno_df):
    """对每个候选亲本预测 6 个性状"""
    snp_names = list(geno_df.columns)
    results = {}
    for trait in TRAITS:
        model_path = os.path.join(MODEL_DIR, f"final_{trait}.pkl")
        prep_path = os.path.join(PREP_DIR, f"final_{trait}.pkl")
        if not os.path.exists(model_path):
            continue
        with open(model_path, "rb") as f:
            model = pickle.load(f)["model"]
        with open(prep_path, "rb") as f:
            pp = pickle.load(f)
        X = geno_df.values.astype(np.float32)
        X = pp["imputer"].transform(X)
        X = pp["scaler"].transform(X)
        results[trait] = model.predict(X)
    df = pd.DataFrame(results, index=geno_df.index)
    df.index.name = "品种"
    return df


def compute_index(pheno_df, weights):
    """综合育种值 I = Σ w * 性状 (各性状先 z-score 标准化)"""
    print("\n" + "=" * 50)
    print("2. 计算综合育种值 (选择指数)")
    print(f"  权重: {weights}")
    z = (pheno_df - pheno_df.mean()) / pheno_df.std()
    I = sum(z[t] * w for t, w in weights.items() if w > 0 and t in pheno_df.columns)
    I = (I - I.min()) / (I.max() - I.min()) * 100   # 归一化到 0-100
    print(f"  综合育种值范围: [{I.min():.1f}, {I.max():.1f}]")
    print(f"  Top5 候选:")
    top5 = I.sort_values(ascending=False).head(5)
    for name, val in top5.items():
        print(f"    {name}: {val:.1f}")
    z_params = {t: (pheno_df[t].mean(), pheno_df[t].std())
                for t in pheno_df.columns}
    return I, z_params


def compute_complementarity(geno_df):
    """用 Ridge β 计算亲本互补性

    关键位点 = 各性状 |β| 最大的 top 5% 位点 (约2500个/性状)
    互补性 = 两亲本在关键位点上"一个0型一个2型"的比例
             (差异大 = 各有所长 = 杂交有互补空间)
    """
    print("\n" + "=" * 50)
    print("3. 计算亲本互补性 (关键位点 = 各性状 |β| top 5%)")

    n_snp = len(geno_df.columns)
    key_mask = np.zeros(n_snp, dtype=bool)
    for trait in TRAITS:
        path = os.path.join(MODEL_DIR, f"final_{trait}.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            beta = np.abs(pickle.load(f)["model"].coef_)
        top_n = max(int(n_snp * 0.05), 100)
        top_idx = np.argsort(beta)[-top_n:]
        key_mask[top_idx] = True

    print(f"  关键位点数: {key_mask.sum()} / {n_snp}")

    # MLM 显著位点加权 (交叉验证过, 富集8-12倍, 权重×2)
    weights = np.ones(key_mask.sum(), dtype=np.float32)
    mlm_csv = "outputs/tassel_gwas/mlm_significant_snps.csv"
    n_mlm = 0
    if os.path.exists(mlm_csv):
        mlm_df = pd.read_csv(mlm_csv)
        mlm_set = set(f"chr{int(r['染色体'])}_{int(r['位置'])}"
                      for _, r in mlm_df.iterrows())
        snp_cols = geno_df.columns[key_mask]
        for i, snp in enumerate(snp_cols):
            if snp in mlm_set:
                weights[i] = 2.0
                n_mlm += 1
    print(f"  MLM 显著位点加权: {n_mlm} 个 (权重×2)")

    Xk = geno_df.values[:, key_mask]           # 关键位点上的基因型
    is0 = (Xk == 0).astype(np.float32)
    is2 = (Xk == 2).astype(np.float32)
    # 加权互补性: diff[i,j] = Σ w_k×[i是0且j是2 或反之] / Σ w_k
    w2 = is2 * weights[None, :]
    w0 = is0 * weights[None, :]
    diff = (is0 @ w2.T + is2 @ w0.T) / weights.sum()
    np.fill_diagonal(diff, 0)
    print(f"  平均互补性: {diff[~np.eye(len(diff), dtype=bool)].mean():.3f}")
    return pd.DataFrame(diff, index=geno_df.index, columns=geno_df.index)


def compute_genetic_distance(geno_df):
    """基因组关系矩阵 G -> 遗传距离"""
    print("\n" + "=" * 50)
    print("4. 计算亲本遗传距离 (防近交)")
    X = geno_df.values.astype(np.float64)
    p = np.nanmean(X / 2.0, axis=0)
    p = np.nan_to_num(p, nan=0.5)
    Xc = X - 2 * p
    Xc = np.nan_to_num(Xc, nan=0.0)
    denom = 2 * np.sum(p * (1 - p))
    G = (Xc @ Xc.T) / denom
    np.fill_diagonal(G, 1.0)
    # 遗传距离 = 2 - 2*亲缘系数 (亲缘系数=G的非对角元素)
    dist = 2 - 2 * G
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    print(f"  平均遗传距离: {dist[~np.eye(len(dist), dtype=bool)].mean():.3f}")
    return pd.DataFrame(dist, index=geno_df.index, columns=geno_df.index)


def score_and_recommend(I, comp, dist, top_n=10):
    """多目标打分, 推荐杂交组合"""
    print("\n" + "=" * 50)
    print(f"5. 推荐杂交组合 (Top {top_n})")
    names = list(I.index)
    I_arr = I.values
    comp_arr = comp.loc[names, names].values
    dist_arr = dist.loc[names, names].values

    records = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            # 后代期望育种值 = 双亲均值
            exp_I = (I_arr[i] + I_arr[j]) / 2
            c = comp_arr[i, j]
            d = dist_arr[i, j]
            # 打分: 育种值 * (1+互补) * (1+距离/2)
            score = exp_I * (1 + c) * (1 + d / 4)
            records.append({
                "亲本A": names[i], "亲本B": names[j],
                "亲本A育种值": round(I_arr[i], 1),
                "亲本B育种值": round(I_arr[j], 1),
                "后代期望": round(exp_I, 1),
                "互补性": round(c, 3),
                "遗传距离": round(d, 3),
                "综合得分": round(score, 1),
            })

    df = pd.DataFrame(records).sort_values("综合得分", ascending=False).head(top_n)
    print(df.to_string(index=False))
    return df


def simulate_cross(geno_df, parent_a, parent_b, n_progeny=100,
                   weights=None, I=None, z_params=None):
    """模拟杂交后代: 孟德尔分离 -> 预测后代表现"""
    print("\n" + "=" * 50)
    print(f"6. 模拟杂交后代: {parent_a} × {parent_b}")

    if parent_a not in geno_df.index or parent_b not in geno_df.index:
        print(f"  [错误] 亲本不在候选列表中")
        return None

    ga = geno_df.loc[parent_a].values.astype(np.float32)
    gb = geno_df.loc[parent_b].values.astype(np.float32)

    # 模拟 n_progeny 个后代: 每位点随机继承 A 或 B 的等位基因
    rng = np.random.default_rng(42)
    n_snp = len(ga)
    progeny = np.zeros((n_progeny, n_snp), dtype=np.float32)
    diff_mask = ga != gb
    for k in range(n_progeny):
        choice = rng.integers(0, 2, size=n_snp)
        progeny[k] = np.where(diff_mask, np.where(choice == 0, ga, gb), ga)

    print(f"  模拟后代数: {n_progeny}")
    prog_df = pd.DataFrame(progeny, columns=geno_df.columns)

    # 预测后代性状
    pheno = predict_all_traits(prog_df)
    print(f"\n  后代 6 性状分布 (均值±标准差):")
    for t in TRAITS:
        if t in pheno.columns:
            print(f"    {t}: {pheno[t].mean():.2f} ± {pheno[t].std():.2f}")

    # 后代选择指数 + 超亲分离
    # 用亲本群体的 z-score 参数标准化后代, 才能和亲本 I 可比
    if weights and I is not None and z_params is not None:
        z = pd.DataFrame(index=pheno.index)
        for t in TRAITS:
            if t in pheno.columns and t in z_params:
                m, s = z_params[t]
                z[t] = (pheno[t] - m) / s if s > 0 else 0.0
        prog_I = sum(z[t] * w for t, w in weights.items() if w > 0)
        prog_I = (prog_I - I.min()) / (I.max() - I.min()) * 100
        parent_max = max(I.get(parent_a, 0), I.get(parent_b, 0))
        trans = (prog_I > parent_max).mean() * 100
        print(f"\n  后代综合育种值: {prog_I.mean():.1f} ± {prog_I.std():.1f}")
        print(f"  亲本较高育种值: {parent_max:.1f}")
        print(f"  超亲分离比例 (优于较高亲本): {trans:.1f}%")
    return pheno


def main():
    parser = argparse.ArgumentParser(description="大豆育种亲本选择")
    parser.add_argument("--weights", nargs="+", default=None,
                        help="性状权重, 如 百粒重=0.3 单株粒重=0.3 荚数=0.2 粒数=0.2")
    parser.add_argument("--top", type=int, default=10, help="推荐组合数")
    parser.add_argument("--simulate", action="store_true",
                        help="对最优组合模拟后代")
    args = parser.parse_args()

    # 解析权重
    weights = DEFAULT_WEIGHTS.copy()
    if args.weights:
        weights = {}
        for item in args.weights:
            k, v = item.split("=")
            weights[k] = float(v)
    # 补全未指定的为 0
    for t in TRAITS:
        weights.setdefault(t, 0.0)

    # ---- 步骤 1-2: 基因型 + 预测 + 综合育种值 ----
    geno = load_genotypes()
    pheno = predict_all_traits(geno)
    I, z_params = compute_index(pheno, weights)

    # ---- 步骤 3: 互补性 ----
    comp = compute_complementarity(geno)

    # ---- 步骤 4: 遗传距离 ----
    dist = compute_genetic_distance(geno)

    # ---- 步骤 5: 推荐组合 ----
    recommend = score_and_recommend(I, comp, dist, top_n=args.top)

    # ---- 步骤 6: 模拟后代 (可选) ----
    if args.simulate and len(recommend) > 0:
        best = recommend.iloc[0]
        simulate_cross(geno, best["亲本A"], best["亲本B"],
                       n_progeny=100, weights=weights, I=I,
                       z_params=z_params)

    # 保存推荐结果
    out = "outputs/recommended_crosses.csv"
    recommend.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n推荐结果已保存: {out}")


if __name__ == "__main__":
    main()
