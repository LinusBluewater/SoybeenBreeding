"""
最终模型固化 v2：600群体 2022+2024 两年合并，全部 6 个性状

依据:
  600群体两年合并后所有性状达到或超过 559 群体:
    百粒重 0.81 (vs 559的0.77), 荚数 0.62, 粒数 0.56,
    单株粒重 0.60, 株高 0.59, 节数 0.56
  -> 统一用 600 群体, 不再混用两群体

表型: 2022+2024 考种表, IQR清洗异常值, 每年5株均值, 两年平均
基因型: geno_600.csv (5万 SNP, AAF编号, 数字对应考种表)

输出:
  outputs/models/final_{trait}.pkl
  outputs/preprocessors/final_{trait}.pkl
  outputs/final_snp_list.txt
  outputs/models/final_summary.csv
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import sklearn
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

GENO = "data/geno_600.csv"

# 两年考种表: (文件, sheet, header行, id列, 性状列映射)
YEARS = [
    ("PhysicData/2022潍坊农科院考种表-给计算机.xlsx", "840份大豆品种", 0, "种植编号",
     {"荚数": "单株荚数", "粒数": "粒数", "百粒重": "百粒重g", "单株粒重": "单株粒重g", "株高": "株高cm", "节数": "主茎节数"}),
    ("PhysicData/2024年潍坊农科院考种-给计算机.xlsx", "山大栽培", 1, "种植编号",
     {"荚数": "单株荚数", "粒数": "粒数", "百粒重": "百粒重", "单株粒重": "单株粒重", "株高": "株高", "节数": "主茎节数"}),
]

TRAITS = ["荚数", "粒数", "百粒重", "单株粒重", "株高", "节数"]
ALPHAS = [1000, 5000, 10000, 50000]
N_FOLDS = 5
SEED = 42

MODEL_DIR = "outputs/models"
PREP_DIR = "outputs/preprocessors"


def clean_iqr(vals):
    """3×IQR 异常值清洗"""
    q1, q3 = np.percentile(vals.dropna(), [25, 75])
    iqr = q3 - q1
    return vals[(vals >= q1 - 3 * iqr) & (vals <= q3 + 3 * iqr)]


def load_year_means(fname, sheet, header, id_col, col_map):
    """读一年考种表: IQR清洗 -> 每品种5株均值"""
    df = pd.read_excel(fname, sheet_name=sheet, header=header)
    nums = df[id_col].ffill().astype(str).str.extract(r"(\d+)", expand=False)

    result = {}
    for trait, col in col_map.items():
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        v_clean = clean_iqr(v)
        mask = v.isin(v_clean) & v.notna()
        sub = pd.DataFrame({"num": nums, "y": v}).loc[mask.values].dropna()
        sub["num"] = sub["num"].astype(int)
        result[trait] = sub.groupby("num")["y"].mean()
    return pd.DataFrame(result)


def cv_ridge(X, y):
    best_alpha, best_r2, best_r = None, -np.inf, 0
    for a in ALPHAS:
        kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
        r2s, rs = [], []
        for tr, te in kf.split(X):
            m = Ridge(alpha=a); m.fit(X[tr], y[tr]); p = m.predict(X[te])
            r2s.append(r2_score(y[te], p)); rs.append(np.corrcoef(y[te], p)[0, 1])
        mr2, mr = np.mean(r2s), np.mean(rs)
        if mr2 > best_r2:
            best_r2, best_r, best_alpha = mr2, mr, a
    return best_r2, best_r, best_alpha


def main():
    print("=" * 50)
    print("最终模型固化 v2 (600群体两年合并)")
    print("=" * 50)

    # ---- 1. 读基因型 ----
    geno = pd.read_csv(GENO, index_col=0)
    print(f"基因型: {geno.shape[0]} 样本 × {geno.shape[1]} SNP")

    # ---- 2. 读两年考种表, 取两年均值 ----
    print("\n读两年考种表...")
    yearly = {}
    for fname, sheet, header, id_col, col_map in YEARS:
        year = "2022" if "2022" in fname else "2024"
        yearly[year] = load_year_means(fname, sheet, header, id_col, col_map)
        print(f"  {year}: {yearly[year].shape[0]} 品种")

    # 两年交集品种
    common = set(yearly["2022"].index) & set(yearly["2024"].index)
    print(f"  两年交集: {len(common)} 品种")

    # 两年均值
    pheno = pd.DataFrame({t: (yearly["2022"][t] + yearly["2024"][t]) / 2
                          for t in TRAITS
                          if t in yearly["2022"].columns and t in yearly["2024"].columns})

    # ---- 3. 品种对齐 (AAF数字 <-> 考种数字) ----
    geno_nums = geno.index.str.extract(r"AAF(\d+)-", expand=False).astype(int).values
    valid_nums = [n for n in common if n in set(geno_nums)]
    print(f"  有基因型+两年表型: {len(valid_nums)} 品种")

    # 保存 SNP 列表
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/final_snp_list.txt", "w") as f:
        f.write("\n".join(geno.columns))

    # ---- 4. 逐性状训练 + 保存 ----
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PREP_DIR, exist_ok=True)

    results = []
    for trait in TRAITS:
        print(f"\n{'=' * 50}")
        print(f"性状: {trait}")

        y_series = pheno.loc[valid_nums, trait].dropna()
        nums = y_series.index
        # 注意: AAF-W 和 AAF-X 数字有重叠, 每个数字只取一个基因型样本
        sample_idx = {}
        for i, n in enumerate(geno_nums):
            if n not in sample_idx:
                sample_idx[n] = i
        idx = [sample_idx[n] for n in nums if n in sample_idx]
        y_series = y_series.loc[[n for n in nums if n in sample_idx]]
        X = geno.iloc[idx].values.astype(np.float32)
        y = y_series.values.astype(np.float32)

        imputer = SimpleImputer(strategy="most_frequent")
        X = imputer.fit_transform(X)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        r2_cv, r_cv, best_alpha = cv_ridge(X, y)
        print(f"  样本: {len(y)}, CV R2={r2_cv:.4f}, CV r={r_cv:.4f}, alpha={best_alpha}")

        model = Ridge(alpha=best_alpha)
        model.fit(X, y)

        model_path = os.path.join(MODEL_DIR, f"final_{trait}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "sklearn_version": sklearn.__version__}, f)

        prep_path = os.path.join(PREP_DIR, f"final_{trait}.pkl")
        with open(prep_path, "wb") as f:
            pickle.dump({"imputer": imputer, "scaler": scaler}, f)
        print(f"  已保存: {model_path}")

        results.append({"性状": trait, "n": len(y), "CV_R2": r2_cv, "CV_r": r_cv, "alpha": best_alpha})

    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(MODEL_DIR, "final_summary.csv"), index=False, encoding="utf-8-sig")
    print(f"\n{'=' * 50}")
    print("最终模型汇总 (600群体 2022+2024 两年合并)")
    print("=" * 50)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
