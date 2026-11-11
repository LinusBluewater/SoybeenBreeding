"""
用 TASSEL 跑 MLM GWAS (混合线性模型, 亲缘矩阵校正群体结构)

流程:
  1. 生成 TASSEL Hapmap 格式基因型 (从 geno_600.csv, 0/1/2 -> 伪碱基)
  2. 生成 TASSEL 表型文件 (品种 × 6性状, 两年均值)
  3. TASSEL: kinship -> MLM (P3D压缩)
  4. 解析 MLM 结果, 输出显著位点

依赖: Tassel5/run_pipeline.bat (项目自带)
"""
import os
import re
import sys
import subprocess
import numpy as np
import pandas as pd

from src.gwas import TasselGWAS

GENO = "data/geno_600.csv"
TASSEL = r"..\Tassel5\run_pipeline.bat"
OUT_DIR = "outputs/tassel_gwas"

# TASSEL 表型名: 用英文 (TASSEL/Java 不支持中文性状名)
TRAITS_MAP = {
    "podnum": ("单株荚数", "单株荚数", None),
    "seednum": ("粒数", "粒数", None),
    "seedwt": ("百粒重g", "百粒重", 100),
    "plantwt": ("单株粒重g", "单株粒重", 500),
    "height": ("株高cm", "株高", None),
    "nodenum": ("主茎节数", "主茎节数", None),
}


def load_two_year_pheno():
    """两年均值表型 (清洗异常值)"""
    pheno = {}
    for fname, sheet, header, prefix in [
        ("PhysicData/2022潍坊农科院考种表-给计算机.xlsx", "840份大豆品种", 0, "AAK"),
        ("PhysicData/2024年潍坊农科院考种-给计算机.xlsx", "山大栽培", 1, "AAP"),
    ]:
        df = pd.read_excel(fname, sheet_name=sheet, header=header)
        nums = df["种植编号"].ffill().astype(str).str.extract(r"(\d+)", expand=False)
        for trait, (col22, col24, clip) in TRAITS_MAP.items():
            col = col22 if header == 0 else col24
            if col not in df.columns:
                continue
            v = pd.to_numeric(df[col], errors="coerce")
            q1, q3 = np.percentile(v.dropna(), [25, 75]); iqr = q3 - q1
            mask = (v >= q1 - 3 * iqr) & (v <= q3 + 3 * iqr)
            sub = pd.DataFrame({"num": nums, "y": v}).loc[mask.values].dropna()
            sub["num"] = sub["num"].astype(int)
            m = sub.groupby("num")["y"].mean()
            pheno.setdefault(trait, {})[prefix] = m

    result = {}
    for trait in TRAITS_MAP:
        if trait not in pheno:
            continue
        common = sorted(set(pheno[trait]["AAK"].index) & set(pheno[trait]["AAP"].index))
        result[trait] = (pheno[trait]["AAK"].loc[common] + pheno[trait]["AAP"].loc[common]) / 2
    return pd.DataFrame(result)


def gen_hapmap(geno_df, out_path):
    """0/1/2 编码 -> TASSEL Hapmap 格式 (0→A, 2→C, NaN→N)"""
    print(f"[1/3] 生成 Hapmap 文件: {out_path}")
    samples = list(geno_df.index)
    header = ["rs#", "alleles", "chrom", "pos", "strand", "assembly#",
              "center", "protLSID", "assayLSID", "panelLSID", "QCcode"] + samples

    with open(out_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for col in geno_df.columns:
            chrom = col.replace("chr", "").split("_")[0]
            pos = col.split("_")[1]
            vals = geno_df[col].values
            # 0->A, 1->R(杂合), 2->C, NaN->N
            letters = np.where(np.isnan(vals), "N",
                               np.where(vals == 0, "A",
                                        np.where(vals == 2, "C", "R")))
            row = [col, "A/C", chrom, pos, "+", "NA", "NA", "NA", "NA", "NA", "NA"]
            row += list(letters)
            f.write("\t".join(row) + "\n")
    print(f"  完成: {len(geno_df.columns)} SNP × {len(samples)} 样本")


def gen_trait_file(pheno_df, geno_df, out_path):
    """TASSEL 表型文件: <Trait> 表头 + -999 缺失 + 品种名与 hapmap 一致"""
    print(f"[2/3] 生成表型文件: {out_path}")
    # 品种名 = hapmap 的原始样本名 (AAF0001-W 等); 每数字取第一个样本
    num_to_name = {}
    for name in geno_df.index:
        num = int(re.match(r"AAF(\d+)-", name).group(1))
        if num not in num_to_name:
            num_to_name[num] = name

    rows = []
    for num, name in num_to_name.items():
        row = [name]
        for trait in TRAITS_MAP:
            if trait in pheno_df.columns and num in pheno_df.index:
                row.append(f"{pheno_df.loc[num, trait]:.4f}")
            else:
                row.append("-999")
        rows.append(row)

    with open(out_path, "w") as f:
        f.write("\t".join(["<Trait>"] + list(TRAITS_MAP.keys())) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"  完成: {len(rows)} 品种 × {len(TRAITS_MAP)} 性状")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    hapmap_path = os.path.join(OUT_DIR, "physic.hmp.txt")
    trait_path = os.path.join(OUT_DIR, "physic_trait.txt")
    kin_path = os.path.join(OUT_DIR, "physic_kin.txt")
    mlm_out = os.path.join(OUT_DIR, "mlm_result")

    # ---- 数据准备 ----
    geno = pd.read_csv(GENO, index_col=0)
    pheno = load_two_year_pheno()
    gen_hapmap(geno, hapmap_path)
    gen_trait_file(pheno, geno, trait_path)

    # ---- TASSEL 运行 ----
    print(f"[3/3] 运行 TASSEL MLM...")
    gwas = TasselGWAS(TASSEL, memory="8g")

    print("\n--- kinship ---")
    gwas.run_kinship(hapmap_path, os.path.join(OUT_DIR, "physic_kin"))

    print("\n--- MLM (P3D 压缩) ---")
    gwas.run_mlm(hapmap_path, trait_path,
                 os.path.join(OUT_DIR, "physic_kin.txt"),
                 mlm_out)

    print(f"\n完成! MLM 结果: {mlm_out}_stats.txt")
    # 列出结果文件
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
