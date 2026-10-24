"""
PhysicData 基因型预处理：Hapmap → 降维 + 编码 → 品种×SNP 矩阵

输入: PhysicData/hapmap.559.hmp.txt.gz (464万 SNP × 559 样本, TASSEL Hapmap 格式)
输出: data/geno_physic.csv (品种 × SNP, 0/1/2 编码)

编码规则:
  alleles = "A/T" (参考=A, 替代=T)
  基因型 == 参考碱基  -> 0 (参考纯合)
  基因型 == 替代碱基  -> 2 (替代纯合)
  基因型 == N         -> NaN (缺失)
  其他(双字母杂合)     -> 1 (杂合)

降维策略: 等距抽样 (每隔 step 个 SNP 取一个), 目标降到 target_n_snps 个
"""
import gzip
import os
import numpy as np
import pandas as pd


# ======================
# 配置
# ======================
HAPMAP_PATH = "PhysicData/hapmap.559.hmp.txt.gz"
OUTPUT_PATH = "data/geno_physic.csv"
TARGET_N_SNPS = 50000          # 降维目标: 5 万个 SNP
TOTAL_SNPS = 4640700           # hapmap 文件的总 SNP 数 (已知)

# 前 11 列是元数据, 第 12 列起是样本基因型
META_COLS = 11


def encode_gts(gts, ref, alt):
    """把一行的基因型值 (碱基字符串列表) 编码成 0/1/2/nan"""
    arr = np.array(gts, dtype=object)
    encoded = np.full(len(gts), np.nan, dtype=np.float32)

    encoded[arr == ref] = 0.0      # 参考纯合
    encoded[arr == alt] = 2.0      # 替代纯合

    # 杂合: 不是参考/替代/缺失的其他值 (双字母如 "A/T" 或 IUPAC 符号)
    valid = np.isin(arr, [ref, alt, "N", "-", ""])
    encoded[~valid] = 1.0

    return encoded


def main():
    print("=" * 50)
    print("PhysicData 基因型预处理")
    print("=" * 50)

    # ---- 1. 读表头, 获取样本名 ----
    print("\n[1/4] 读取表头...")
    with gzip.open(HAPMAP_PATH, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")

    samples = header[META_COLS:]
    n_samples = len(samples)
    print(f"  样本数: {n_samples}")
    print(f"  样本名示例: {samples[:5]}")

    # ---- 2. 等距抽样 + 编码 ----
    step = TOTAL_SNPS // TARGET_N_SNPS
    print(f"\n[2/4] 等距抽样 (每隔 {step} 个 SNP 取一个)...")
    print(f"  目标: {TARGET_N_SNPS} 个 SNP")

    snp_names = []
    genotype_matrix = []   # 每行是一个 SNP 的 559 个编码值

    with gzip.open(HAPMAP_PATH, "rt") as f:
        f.readline()  # 跳过表头
        for i, line in enumerate(f):
            if i % step != 0:
                continue
            fields = line.rstrip("\n").split("\t")
            snp_name = fields[0]            # rs#, 如 S01_10679
            alleles = fields[1]             # 如 "A/T"
            ref, alt = alleles.split("/")
            gts = fields[META_COLS:]        # 559 个基因型值

            encoded = encode_gts(gts, ref, alt)
            snp_names.append(snp_name)
            genotype_matrix.append(encoded)

            # 进度提示
            if len(snp_names) % 5000 == 0:
                print(f"  已处理 {len(snp_names)} 个 SNP...")

    n_snps = len(snp_names)
    print(f"  抽样完成: {n_snps} 个 SNP")

    # ---- 3. 转置成 品种 × SNP 矩阵 ----
    print("\n[3/4] 转置成品种×SNP 矩阵...")
    # genotype_matrix: list[snps], 每个是长度 n_samples 的数组
    # 转置 -> (n_samples, n_snps)
    mat = np.array(genotype_matrix, dtype=np.float32).T  # (559, 50000)
    print(f"  矩阵形状: {mat.shape} (品种 × SNP)")

    geno_df = pd.DataFrame(mat, index=samples, columns=snp_names)
    geno_df.index.name = "strain"

    # ---- 4. 保存 ----
    print("\n[4/4] 保存...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    geno_df.to_csv(OUTPUT_PATH)
    print(f"  已保存到: {OUTPUT_PATH}")

    # 统计
    n_missing = np.isnan(mat).sum()
    print(f"\n  缺失率: {n_missing / mat.size * 100:.2f}% "
          f"({n_missing}/{mat.size})")
    print(f"  各基因型频率:")
    for code, name in [(0.0, '0(参考纯合)'), (1.0, '1(杂合)'), (2.0, '2(替代纯合)')]:
        cnt = (mat == code).sum()
        print(f"    {name}: {cnt/mat.size*100:.1f}%")
    print("\n预处理完成!")


if __name__ == "__main__":
    main()
