import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM


def extract_blup(
    pheno_df,
    trait,
    env_col="environ",
    group_col="strain",
    method="bfgs",
    maxiter=1000,
    min_envs_per_strain=2,
    verbose=True
):
    """
    用混合线性模型从多环境表型数据提取育种值 (BLUP)。

    模型:  trait ~ C(env_col) + (1 | group_col)
           固定效应: 环境主效应（被吸收，不进入育种值）
           随机效应: 品种 → BLUP 即每个品种的育种值偏差

    参数
    ----
    pheno_df : pd.DataFrame
        至少包含 group_col, env_col, trait 三列
    trait : str
        目标性状列名（若为 Python 关键字如 "yield"，会自动重命名）
    env_col : str
        环境列名（固定效应）
    group_col : str
        分组列名（随机效应，通常是品种 strain）
    method : str
        优化器：bfgs / lbfgs / powell
    min_envs_per_strain : int
        品种至少需出现在 N 个环境中，否则剔除（BLUP 不稳定）
    verbose : bool
        是否打印进度

    返回
    ----
    result : dict
        - "blup_df": DataFrame[strain, blup, gebv, n_envs, n_obs]
        - "grand_mean": 总体均值
        - "var_comp": dict(sigma2_u, sigma2_e, h2)  遗传力
        - "model": 拟合好的 MixedLMResults（可复用）
    """
    warnings.filterwarnings("ignore")

    # ---- 1. 准备数据 ----
    keep = [group_col, env_col, trait]
    df = pheno_df[keep].copy().dropna(subset=[trait])

    # 避开 Python 关键字（patsy 公式里 yield 是关键字）
    formula_trait = trait
    rename_map = {}
    if trait in ("yield", "class", "return", "import"):
        formula_trait = trait + "_y"
        rename_map = {trait: formula_trait}
        df = df.rename(columns=rename_map)

    # 品种至少出现在 min_envs_per_strain 个环境中
    env_count = df.groupby(group_col)[env_col].nunique()
    keep_strains = env_count[env_count >= min_envs_per_strain].index
    df = df[df[group_col].isin(keep_strains)].copy()

    if verbose:
        print(f"  [BLUP] 输入: {len(df)} 条记录, "
              f"{df[group_col].nunique()} 个{group_col}, "
              f"{df[env_col].nunique()} 个{env_col}")
        if len(rename_map):
            print(f"  [BLUP] 注意: '{trait}' 是关键字, 已重命名为 '{formula_trait}'")

    # ---- 2. 拟合混合线性模型 ----
    formula = f"{formula_trait} ~ C({env_col})"
    model = MixedLM.from_formula(formula, data=df, groups=df[group_col])
    mdf = model.fit(method=method, maxiter=maxiter, disp=False)

    # ---- 3. 提取 BLUP ----
    # random_effects 返回 dict: {strain: Series([blup_value])}
    re = mdf.random_effects
    blup = pd.Series({k: float(v.iloc[0]) for k, v in re.items()})
    blup.name = "blup"

    # 品种元信息
    meta = df.groupby(group_col).agg(
        n_envs=(env_col, "nunique"),
        n_obs=(formula_trait, "size")
    )

    blup_df = pd.DataFrame({"blup": blup}).join(meta)

    # ---- 4. 方差分量与遗传力 ----
    sigma2_u = float(mdf.cov_re.iloc[0, 0])     # 品种间方差
    sigma2_e = float(mdf.scale)                 # 残差方差
    grand_mean = float(df[formula_trait].mean())
    n_per = df.groupby(group_col).size().mean() # 每品种平均重复数
    h2 = sigma2_u / (sigma2_u + sigma2_e / n_per)

    # GEBV = 总均值 + BLUP 偏差（回到实际表型单位）
    blup_df["gebv"] = grand_mean + blup_df["blup"]

    if verbose:
        print(f"  [BLUP] sigma2_u (genetic)  = {sigma2_u:.0f}")
        print(f"  [BLUP] sigma2_e (residual) = {sigma2_e:.0f}")
        print(f"  [BLUP] heritability h2 ~ {h2:.3f}")
        print(f"  [BLUP] grand mean mu = {grand_mean:.1f}")
        print(f"  [BLUP] GEBV range: "
              f"[{blup_df['gebv'].min():.1f}, {blup_df['gebv'].max():.1f}], "
              f"std = {blup_df['gebv'].std():.1f}")

    return {
        "blup_df": blup_df,
        "grand_mean": grand_mean,
        "var_comp": {"sigma2_u": sigma2_u, "sigma2_e": sigma2_e, "h2": h2},
        "model": mdf,
    }


def save_blup(result, trait, out_dir):
    """保存 BLUP 结果到 csv"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"blup_{trait}.csv")
    result["blup_df"].to_csv(path)
    return path
