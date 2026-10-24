"""
多年 BLUP 合并训练：利用 2022/2023/2024 三年考种表提升预测精度

三年品种交集: 750 个 (同数字=同材料, 说明文件规则)

流程:
  1. 读三年考种表, 统一性状列名, 每年每品种取 5 株均值(清洗异常值)
  2. 三种表型目标:
     A. 单年(2022)均值    - 之前的结果
     B. 多年简单均值      - 三年均值的平均
     C. 多年 BLUP 育种值  - 性状~C(年份)+(1|品种)
  3. 分别用 600群体基因型训练, 对比 CV r

核心问题: 多年数据能否提升预测精度?
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

GENO_600 = "data/geno_600.csv"
# 注意: 2023 年测量定义与其他年份不同 (跨年相关 -0.01~0.22), 已排除
# 2022-2024 跨年相关高 (百粒重0.89/株高0.62/节数0.43), 定义一致, 可合并
YEAR_FILES = [
    ("PhysicData/2022潍坊农科院考种表-给计算机.xlsx", "840份大豆品种", "种植编号", "2022",
     {"荚数": "单株荚数", "粒数": "粒数", "百粒重": "百粒重g", "单株粒重": "单株粒重g", "株高": "株高cm", "节数": "主茎节数"}),
    ("PhysicData/2024年潍坊农科院考种-给计算机.xlsx", "山大栽培", "种植编号", "2024",
     {"荚数": "单株荚数", "粒数": "粒数", "百粒重": "百粒重", "单株粒重": "单株粒重", "株高": "株高", "节数": "主茎节数"}),
]

CLIP_HIGH = {"百粒重": 100, "单株粒重": 500}   # 异常值上限

ALPHAS = [1000, 5000, 10000, 50000]
N_FOLDS = 5
SEED = 42


def load_year(fname, sheet, id_col, year, col_map):
    """读一年的考种表, 返回 (品种数字, 性状均值 DataFrame)"""
    header = 1 if year == "2024" else 0
    df = pd.read_excel(fname, sheet_name=sheet, header=header)
    df = df.rename(columns={id_col: "品种"})
    df["品种"] = df["品种"].ffill()   # 合并单元格向前填充
    df["品种"] = df["品种"].astype(str).str.extract(r"(\d+)", expand=False)
    df = df.dropna(subset=["品种"])
    df["品种"] = df["品种"].astype(int)

    result = {}
    for trait, col in col_map.items():
        if col not in df.columns:
            continue
        sub = df[["品种", col]].rename(columns={col: "y"})
        sub["y"] = pd.to_numeric(sub["y"], errors="coerce")
        sub = sub.dropna()
        if trait in CLIP_HIGH:
            sub = sub[sub["y"] <= CLIP_HIGH[trait]]
        result[trait] = sub.groupby("品种")["y"].mean()   # 5株均值

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
    return best_r2, best_r


def main():
    print("=" * 50)
    print("1. 读三年考种表")
    years = {}
    for fname, sheet, id_col, year, col_map in YEAR_FILES:
        years[year] = load_year(fname, sheet, id_col, year, col_map)
        print(f"  {year}: {years[year].shape[0]} 品种")

    # 两年交集 (2022 + 2024)
    common = set(years["2022"].index)
    for y in ["2024"]:
        common &= set(years[y].index)
    print(f"  两年交集: {len(common)} 个品种")

    # ---- 基因型 ----
    geno = pd.read_csv(GENO_600, index_col=0)
    geno_nums = geno.index.str.extract(r"AAF(\d+)-", expand=False).astype(int).values
    # 有基因型 + 三年表型的品种
    common_geno = [n for n in common if n in set(geno_nums)]
    print(f"  有基因型+三年表型: {len(common_geno)} 个品种")

    print("\n" + "=" * 50)
    print("2. 对比: 单年 vs 多年均值 vs 多年BLUP")
    print("=" * 50)
    print(f"{'性状':10s} {'2022单年':>10s} {'多年均值':>10s} {'多年BLUP':>10s} {'BLUP提升':>10s}")

    summary = []
    for trait in ["荚数", "粒数", "百粒重", "单株粒重", "株高", "节数"]:
        # 提取两年该性状
        yearly = {}
        for y in ["2022", "2024"]:
            if trait in years[y].columns:
                yearly[y] = years[y][trait]

        # 长表: 品种 × 年份
        rows = []
        for y, series in yearly.items():
            for num, val in series.items():
                if num in common_geno:
                    rows.append({"品种": num, "年份": y, "y": val})
        long_df = pd.DataFrame(rows).dropna(subset=["y"])

        # A. 单年 2022
        y2022 = long_df[long_df["年份"] == "2022"].set_index("品种")["y"]

        # B. 多年简单均值
        y_multi_mean = long_df.groupby("品种")["y"].mean()

        # C. 多年 BLUP (年份固定 + 品种随机)
        md = MixedLM.from_formula("y ~ C(年份)", data=long_df, groups=long_df["品种"])
        mdf = md.fit(method="bfgs", disp=False)
        blup = pd.Series({k: float(v.iloc[0]) for k, v in mdf.random_effects.items()})

        # 基因型对齐
        def train_eval(target):
            nums = [n for n in geno_nums if n in target.index]
            idx = [i for i, n in enumerate(geno_nums) if n in target.index]
            X = geno.iloc[idx].values.astype(np.float32)
            X = SimpleImputer(strategy="most_frequent").fit_transform(X)
            X = StandardScaler().fit_transform(X)
            y = target.loc[[n for n in geno_nums if n in target.index]].values
            return cv_ridge(X, y)

        r2_a, r_a = train_eval(y2022)
        r2_b, r_b = train_eval(y_multi_mean)
        r2_c, r_c = train_eval(blup)
        delta = r_c - r_a
        print(f"{trait:10s} {r_a:>10.4f} {r_b:>10.4f} {r_c:>10.4f} {delta:>+10.4f}")

        summary.append({"性状": trait, "2022单年": r_a, "多年均值": r_b,
                        "多年BLUP": r_c, "BLUP提升": delta, "n": len(common_geno)})

    print("\n" + "=" * 50)
    print("汇总")
    print("=" * 50)
    df = pd.DataFrame(summary)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
