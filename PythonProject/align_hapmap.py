"""
从 hapmap 提取 VCF 抽样位置的基因型（SNP 对齐）

问题: 两个预处理脚本用不同的等距抽样步长, 抽样相位错开, 共同 SNP 只有 1%
方案: 以 geno_600.csv (VCF 等距抽样) 的 SNP 位置为准, 从 hapmap 提取这些位置

输入: data/geno_600.csv 的列名 (chr1_742 位置集合)
      PhysicData/hapmap.559.hmp.txt.gz (流式)
输出: data/geno_559_aligned.csv (559样本 × 共同SNP)
"""
import gzip
import re
import os
import numpy as np
import pandas as pd

GENO_600 = "data/geno_600.csv"
HAPMAP = "PhysicData/hapmap.559.hmp.txt.gz"
OUTPUT = "data/geno_559_aligned.csv"
META_COLS = 11


def encode_gts(gts, ref, alt):
    """碱基 -> 0/1/2/nan (同 preprocess_geno.py)"""
    arr = np.array(gts, dtype=object)
    encoded = np.full(len(gts), np.nan, dtype=np.float32)
    encoded[arr == ref] = 0.0
    encoded[arr == alt] = 2.0
    valid = np.isin(arr, [ref, alt, "N", "-", ""])
    encoded[~valid] = 1.0
    return encoded


def main():
    print("=" * 50)
    print("hapmap SNP 对齐")
    print("=" * 50)

    # ---- 1. 目标 SNP 位置集合 (来自 VCF 抽样) ----
    geno600 = pd.read_csv(GENO_600, index_col=0, nrows=1)
    target = {}
    for name in geno600.columns:
        m = re.match(r"chr(\d+)_(\d+)", name)
        if m:
            target[(int(m.group(1)), int(m.group(2)))] = name
    print(f"[1/3] 目标 SNP 位置: {len(target)} 个 (来自 VCF 抽样)")

    # ---- 2. 流式读 hapmap, 提取目标位置 ----
    print("[2/3] 流式读取 hapmap, 提取目标位置...")
    found = {}   # (chrom,pos) -> 编码数组

    with gzip.open(HAPMAP, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        samples = header[META_COLS:]
        n_samples = len(samples)

        for line in f:
            fields = line.rstrip("\n").split("\t")
            if not fields[2].isdigit():   # 跳过 SCAFFOLD 等非数字染色体
                continue
            chrom = int(fields[2])     # '01' -> 1
            pos = int(fields[3])
            key = (chrom, pos)
            if key in target:
                alleles = fields[1]
                ref, alt = alleles.split("/")
                gts = fields[META_COLS:]
                found[key] = encode_gts(gts, ref, alt)
                if len(found) % 5000 == 0:
                    print(f"  已找到 {len(found)}/{len(target)} ...")

    print(f"  找到 {len(found)}/{len(target)} 个共同 SNP ({len(found)/len(target)*100:.1f}%)")

    # ---- 3. 保存 (按 geno_600 的列顺序) ----
    print("[3/3] 保存对齐后的 559 基因型...")
    ordered_keys = [(int(m.group(1)), int(m.group(2)))
                    for name in geno600.columns
                    if (m := re.match(r"chr(\d+)_(\d+)", name))]
    common_keys = [k for k in ordered_keys if k in found]

    mat = np.array([found[k] for k in common_keys], dtype=np.float32).T
    snp_names = [target[k] for k in common_keys]
    geno559_aligned = pd.DataFrame(mat, index=samples, columns=snp_names)
    geno559_aligned.index.name = "strain"

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    geno559_aligned.to_csv(OUTPUT)
    print(f"  已保存: {OUTPUT} ({geno559_aligned.shape})")


if __name__ == "__main__":
    main()
