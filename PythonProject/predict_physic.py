"""
统一预测脚本 v2：直接接受标准 VCF（重测序输出）或基因型矩阵

用法:
  python predict_physic.py 新品种.vcf.gz       # VCF 输入 (推荐, 重测序标准输出)
  python predict_physic.py 新品种.vcf          # 未压缩 VCF
  python predict_physic.py 新品种.csv          # 基因型矩阵 (兼容旧格式)

VCF 模式: 自动按 染色体+位置 提取训练用的 SNP 位点, 无需手动处理
  - 新品种测序比对到 Wm82.gnm2 后产出的 VCF 直接可用
  - 缺失的训练位点自动填 0
  - 参考基因组必须与训练一致 (Wm82.gnm2)

CSV 模式: 列名需为 chr{染色体}_{位置} 格式 (如 chr1_742)
"""
import os
import sys
import gzip
import pickle
import numpy as np
import pandas as pd
import sklearn

TRAITS = ["百粒重", "荚数", "粒数", "单株粒重", "株高", "节数"]
MODEL_DIR = "outputs/models"
PREP_DIR = "outputs/preprocessors"
SNP_LIST = "outputs/final_snp_list.txt"

_checked = set()


def load_trait(trait):
    """加载性状的模型和预处理器"""
    model_path = os.path.join(MODEL_DIR, f"final_{trait}.pkl")
    prep_path = os.path.join(PREP_DIR, f"final_{trait}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型未找到: {model_path}\n请先运行 train_final.py")
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    model = obj["model"] if isinstance(obj, dict) else obj
    train_ver = obj.get("sklearn_version") if isinstance(obj, dict) else None
    cur = sklearn.__version__
    if train_ver and train_ver != cur and trait not in _checked:
        _checked.add(trait)
        print(f"  [警告] '{trait}' 模型 sklearn {train_ver}, 当前 {cur}")
    with open(prep_path, "rb") as f:
        pp = pickle.load(f)
    return model, pp["imputer"], pp["scaler"]


def load_snp_list():
    """训练 SNP 列表: (chrX_pos 名称, (chr, pos))"""
    pairs = []
    for line in open(SNP_LIST):
        name = line.strip()
        if not name:
            continue
        parts = name.replace("chr", "").split("_")
        pairs.append((name, (int(parts[0]), int(parts[1]))))
    return pairs


def encode_gt(gt):
    """GT 0/0->0, 0/1->1, 1/1->2, ./.->nan"""
    if gt.startswith("./") or gt.startswith(".|"):
        return np.nan
    a, b = gt[0], gt[2]
    if a == "." or b == ".":
        return np.nan
    return float(int(a) + int(b))


def read_vcf(vcf_path, snp_pairs):
    """从 VCF 提取训练 SNP 位点的基因型矩阵 (支持 .vcf / .vcf.gz / .vcf.zip)"""
    import zipfile
    snp_target = {pos: name for name, pos in snp_pairs}

    samples = None
    found = {}   # name -> 基因型数组

    # 打开文件句柄
    if vcf_path.endswith(".zip"):
        z = zipfile.ZipFile(vcf_path)
        fh = z.open(z.namelist()[0])
        lines = (line.decode("utf-8", "ignore") for line in fh)
    elif vcf_path.endswith(".gz"):
        fh = gzip.open(vcf_path, "rt")
        lines = fh
    else:
        fh = open(vcf_path, "rt")
        lines = fh

    try:
        for line in lines:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                for name in snp_target.values():
                    found[name] = np.full(len(samples), np.nan, dtype=np.float32)
                continue
            if samples is None:
                continue
            fields = line.rstrip("\n").split("\t")
            chrom, pos = int(fields[0]), int(fields[1])
            key = (chrom, pos)
            if key in snp_target:
                name = snp_target[key]
                gts = [fields[9 + i].split(":")[0] for i in range(len(samples))]
                found[name] = np.array([encode_gt(g) for g in gts], dtype=np.float32)
    finally:
        fh.close()

    if samples is None:
        raise ValueError("VCF 无 #CHROM 表头, 文件格式错误")

    mat = np.array([found[name] for name, _ in snp_pairs], dtype=np.float32).T
    n_missing_snp = sum(1 for name, _ in snp_pairs
                        if np.isnan(found[name]).all())
    print(f"  VCF 样本数: {len(samples)}")
    print(f"  训练 SNP {len(snp_pairs)} 个, VCF 中命中 "
          f"{len(snp_pairs) - n_missing_snp} 个, 缺失 {n_missing_snp} 个(填0)")
    return pd.DataFrame(mat, index=samples, columns=[n for n, _ in snp_pairs])


def read_matrix(csv_path, snp_names):
    """CSV 矩阵输入 (兼容旧格式)"""
    geno = pd.read_csv(csv_path, index_col=0)
    missing = set(snp_names) - set(geno.columns)
    if missing:
        print(f"  [警告] 缺失 {len(missing)} 个 SNP, 填 0")
        for s in missing:
            geno[s] = 0
    return geno[snp_names]


def main():
    if len(sys.argv) < 2:
        print("用法: python predict_physic.py <新品种.vcf[.gz] | 新品种.csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "predictions_physic.csv"

    snp_pairs = load_snp_list()
    snp_names = [n for n, _ in snp_pairs]

    # ---- 1. 读取基因型 (自动识别格式) ----
    print("=" * 50)
    print(f"1. 加载新品种基因型: {input_path}")
    if input_path.endswith((".vcf", ".vcf.gz", ".vcf.zip", ".zip")):
        geno = read_vcf(input_path, snp_pairs)
    else:
        geno = read_matrix(input_path, snp_names)
        print(f"  {geno.shape[0]} 个品种 × {geno.shape[1]} 个 SNP")
    print(f"  SNP 已对齐: {geno.shape[1]} 个")

    # ---- 2. 逐性状预测 ----
    print(f"\n2. 预测 {len(TRAITS)} 个性状...")
    results = {"品种": geno.index.tolist()}
    for trait in TRAITS:
        try:
            model, imputer, scaler = load_trait(trait)
            X = geno.values.astype(np.float32)
            X = imputer.transform(X)
            X = scaler.transform(X)
            pred = model.predict(X)
            results[trait] = np.round(pred, 2)
            print(f"  {trait:8s} ✓  范围 [{pred.min():.2f}, {pred.max():.2f}]")
        except FileNotFoundError as e:
            print(f"  {trait:8s} ⊘  跳过 ({e})")

    # ---- 3. 输出 ----
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n3. 预测完成: {out_path}")
    print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
