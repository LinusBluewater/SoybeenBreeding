"""
部署预测脚本：根据新品种基因型预测育种值 (GEBV)。

用法:
  python predict.py <genotype_file.csv> [--traits yield protein oil]
                                                    ^ 不指定则预测全部已训练性状

输入:
  genotype_file.csv: 品种 × SNP 矩阵
    - 第0列为品种名 (index)
    - 列为 SNP 标记 (与训练数据相同命名，如 Gm01_10510009_G_T)
    - 值为 0 (参考型纯合) / 2 (替代型纯合) / NA (缺失)

输出:
  predictions.csv: 每个品种的各性状 GEBV 预测值

文件依赖 (需预先训练):
  outputs/models/ridge_{trait}.pkl          — Ridge 模型
  outputs/preprocessors/preprocessor_{trait}.pkl — imputer + scaler
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd


# ======================
# 可预测的性状 (由已训练的模型决定)
# ======================
TRAITS = ["height", "R8", "planting", "flower", "maturity",
          "lodging", "yield", "protein", "oil", "size"]

MODEL_DIR = "outputs/models"
PREP_DIR = "outputs/preprocessors"


def load_model_and_preprocessor(trait):
    """加载指定性状的模型和预处理器"""
    model_path = os.path.join(MODEL_DIR, f"ridge_{trait}.pkl")
    prep_path = os.path.join(PREP_DIR, f"preprocessor_{trait}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型未找到: {model_path}\n请先运行 train_ridge.py --target {trait}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(prep_path, "rb") as f:
        pp = pickle.load(f)

    return model, pp["imputer"], pp["scaler"]


def predict(geno_path, traits, out_path="predictions.csv"):
    """主预测流程"""
    # 1. 加载输入基因型
    print(f"[1/4] 加载基因型: {geno_path}")
    geno_new = pd.read_csv(geno_path, index_col=0)
    n_strains, n_snps = geno_new.shape
    print(f"  {n_strains} 个品种, {n_snps} 个 SNP")

    # 2. 加载第一个模型的预处理器，确认 SNP 列对齐
    _, imputer_ref, scaler_ref = load_model_and_preprocessor(traits[0])
    # 用训练时的 imputer 填充
    X = imputer_ref.transform(geno_new.values.astype(np.float32))
    X = scaler_ref.transform(X)
    print(f"  预处理完成: {X.shape}")

    # 3. 逐个性状预测
    print(f"[2/4] 预测 {len(traits)} 个性状...")
    results = {"strain": geno_new.index.tolist()}

    for trait in traits:
        try:
            model, imputer, scaler = load_model_and_preprocessor(trait)
            X_t = imputer.transform(geno_new.values.astype(np.float32))
            X_t = scaler.transform(X_t)
            pred = model.predict(X_t)
            results[trait] = pred.round(4)
            print(f"  {trait:>12s} ✓  范围 [{pred.min():.2f}, {pred.max():.2f}]")
        except FileNotFoundError:
            print(f"  {trait:>12s} ⊘  跳过 (模型不存在)")

    # 4. 输出
    print(f"[3/4] 保存预测结果 -> {out_path}")
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"[4/4] 完成! 输出 {df.shape[0]} 个品种 × {df.shape[1]-1} 个性状")
    return df


def main():
    parser = argparse.ArgumentParser(description="rrBLUP 育种值预测")
    parser.add_argument("input", type=str, nargs="?", default="",
                        help="基因型 CSV 文件路径 (--list 时可不传)")
    parser.add_argument("--traits", type=str, nargs="+", default=None,
                        help=f"预测的性状列表 (默认: 全部已训练)")
    parser.add_argument("--out", type=str, default="predictions.csv",
                        help="输出文件路径")
    parser.add_argument("--list", action="store_true",
                        help="列出可用性状并退出")
    args = parser.parse_args()

    if args.list:
        available = [t for t in TRAITS
                     if os.path.exists(os.path.join(MODEL_DIR, f"ridge_{t}.pkl"))]
        print("可用性状:", ", ".join(available))
        print("已训练:", len(available), "个")
        return

    if not os.path.exists(args.input):
        print(f"错误: 文件不存在: {args.input}")
        sys.exit(1)

    traits = args.traits if args.traits else [
        t for t in TRAITS
        if os.path.exists(os.path.join(MODEL_DIR, f"ridge_{t}.pkl"))
    ]
    print(f"预测性状: {traits}")

    predict(args.input, traits, args.out)


if __name__ == "__main__":
    main()
