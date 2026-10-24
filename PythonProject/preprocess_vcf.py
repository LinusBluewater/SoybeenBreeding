"""
VCF 基因型预处理：提取 AAF 样本的基因型，等距抽样 + 编码

输入: PhysicData/600.maf0.05.miss0.1.vcf.zip (560万 SNP × 600 样本)
输出: data/geno_600.csv (AAF样本 × SNP, 0/1/2 编码)

编码规则 (GT 格式):
  0/0 -> 0 (参考纯合)
  0/1 -> 1 (杂合)
  1/1 -> 2 (替代纯合)
  ./. -> NaN (缺失)

只提取 AAF-W 和 AAF-X 样本（对应考种表 AAK 品种，数字相同）
"""
import zipfile
import os
import numpy as np
import pandas as pd


VCF_ZIP = "PhysicData/600.maf0.05.miss0.1.vcf.zip"
OUTPUT = "data/geno_600.csv"
TARGET_N_SNPS = 50000
TOTAL_SNPS = 5600946   # 已知


def encode_gt(gt):
    """GT 格式 0/0, 0/1, 1/1, ./. -> 0, 1, 2, nan"""
    if gt.startswith("./") or gt.startswith(".|"):
        return np.nan
    # 提取两个等位基因数字
    a = gt[0]
    b = gt[2]
    if a == "." or b == ".":
        return np.nan
    return float(int(a) + int(b))   # 0/0->0, 0/1->1, 1/1->2


def main():
    print("=" * 50)
    print("VCF 基因型预处理")
    print("=" * 50)

    z = zipfile.ZipFile(VCF_ZIP)
    vcf_name = z.namelist()[0]
    f = z.open(vcf_name)

    # ---- 1. 读表头，确定样本 ----
    print("\n[1/4] 读取表头...")
    header = None
    for line in f:
        line = line.decode("utf-8", errors="ignore")
        if line.startswith("#CHROM"):
            header = line.rstrip("\n").split("\t")
            break

    all_samples = header[9:]   # 前 9 列是标准字段
    print(f"  VCF 总样本数: {len(all_samples)}")

    # 只提取 AAF-W 和 AAF-X 样本
    target_idx = []
    target_names = []
    for i, s in enumerate(all_samples):
        if s.startswith("AAF") and (s.endswith("-W") or s.endswith("-X")):
            target_idx.append(i)
            target_names.append(s)
    print(f"  提取 AAF 样本: {len(target_names)} 个")
    print(f"  示例: {target_names[:5]}...{target_names[-3:]}")

    # ---- 2. 流式等距抽样 + 编码 ----
    step = TOTAL_SNPS // TARGET_N_SNPS
    print(f"\n[2/4] 等距抽样 (每隔 {step} 个 SNP 取一个)...")

    snp_names = []
    genotype_matrix = []   # 每个元素是 426 个 AAF 样本的编码值

    for i, line in enumerate(f):
        if i % step != 0:
            continue
        line = line.decode("utf-8", errors="ignore")
        fields = line.rstrip("\n").split("\t")
        chrom, pos = fields[0], fields[1]
        snp_name = f"chr{chrom}_{pos}"
        snp_names.append(snp_name)

        # 提取 AAF 样本的 GT 值，编码
        gts = [fields[9 + idx].split(":")[0] for idx in target_idx]  # GT 是第一个字段
        encoded = np.array([encode_gt(g) for g in gts], dtype=np.float32)
        genotype_matrix.append(encoded)

        if len(snp_names) % 5000 == 0:
            print(f"  已处理 {len(snp_names)} 个 SNP...")

    n_snps = len(snp_names)
    print(f"  抽样完成: {n_snps} 个 SNP")

    # ---- 3. 转置成 样本 × SNP 矩阵 ----
    print("\n[3/4] 转置成样本×SNP 矩阵...")
    mat = np.array(genotype_matrix, dtype=np.float32).T
    geno_df = pd.DataFrame(mat, index=target_names, columns=snp_names)
    geno_df.index.name = "sample"
    print(f"  矩阵形状: {geno_df.shape}")

    # ---- 4. 保存 ----
    print("\n[4/4] 保存...")
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    geno_df.to_csv(OUTPUT)
    print(f"  已保存到: {OUTPUT}")

    n_missing = np.isnan(mat).sum()
    print(f"\n  缺失率: {n_missing / mat.size * 100:.2f}%")
    print(f"  基因型频率: 0={np.mean(mat==0)*100:.1f}%, "
          f"1={np.mean(mat==1)*100:.1f}%, 2={np.mean(mat==2)*100:.1f}%")
    print("\n预处理完成!")


if __name__ == "__main__":
    main()
