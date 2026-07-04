import os
import subprocess


class TasselGWAS:

    def __init__(self, tassel_path, memory="8g"):
        self.tassel_path = tassel_path
        self.memory = memory

    # =========================
    # run command
    # =========================
    def run_cmd(self, cmd):
        print("\n[GWAS CMD]")
        print(" ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError("TASSEL failed")

    # =========================
    # SNP filtering
    # =========================
    def filter_snp(self, genotype_file, output_prefix, maf=0.05, min_count=150):

        cmd = [
            self.tassel_path,
            f"-Xmx{self.memory}",
            "-fork1",
            "-h", genotype_file,
            "-filterAlign",
            "-filterAlignMinFreq", str(maf),
            "-filterAlignMinCount", str(min_count),
            "-export", output_prefix,
            "-exportType", "Hapmap"
        ]

        self.run_cmd(cmd)
        return output_prefix + "1.hmp.txt"

    # =========================
    # PCA
    # =========================
    def run_pca(self, filtered_file, output_prefix):

        cmd = [
            self.tassel_path,
            f"-Xmx{self.memory}",
            "-fork1",
            "-h", filtered_file,
            "-PrincipalComponentsPlugin",
            "-endPlugin",
            "-export", output_prefix
        ]

        self.run_cmd(cmd)
        return output_prefix + "1.txt"

    # =========================
    # Kinship
    # =========================
    def run_kinship(self, filtered_file, output_prefix):

        cmd = [
            self.tassel_path,
            f"-Xmx{self.memory}",
            "-fork1",
            "-h", filtered_file,
            "-KinshipPlugin",
            "-endPlugin",
            "-export", output_prefix
        ]

        self.run_cmd(cmd)
        return output_prefix + ".txt"

    # =========================
    # MLM GWAS
    # =========================
    def run_mlm(self, filtered_file, pheno_file, kinship_file,
                output_file, pca_file=None):

        if pca_file is None:

            cmd = [
                self.tassel_path,
                f"-Xmx{self.memory}",
                "-fork1", "-h", filtered_file,
                "-fork2", "-r", pheno_file,
                "-fork3", "-k", kinship_file,
                "-combine4",
                "-input1", "-input2",
                "-intersect",
                "-combine5",
                "-input4", "-input3",
                "-mlm",
                "-mlmVarCompEst", "P3D",
                "-mlmCompressionLevel", "Optimum",
                "-mlmOutputFile", output_file
            ]

        else:

            cmd = [
                self.tassel_path,
                f"-Xmx{self.memory}",
                "-fork1", "-h", filtered_file,
                "-fork2", "-r", pheno_file,
                "-fork3", "-q", pca_file,
                "-fork4", "-k", kinship_file,
                "-combine5",
                "-input1", "-input2", "-input3",
                "-intersect",
                "-combine6",
                "-input5", "-input4",
                "-mlm",
                "-mlmVarCompEst", "P3D",
                "-mlmCompressionLevel", "Optimum",
                "-mlmOutputFile", output_file
            ]

        self.run_cmd(cmd)
        return output_file