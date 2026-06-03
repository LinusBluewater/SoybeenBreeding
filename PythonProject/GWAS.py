# =========================================================
# TASSEL5 GWAS Pipeline
# 1. SNP过滤
# 2. PCA分析
# 3. Kinship矩阵
# 4. MLM GWAS
# =========================================================

import os
import subprocess


class TasselGWASPipeline:

    def __init__(
            self,
            tassel_path=r"..\TASSEL5\run_pipeline.bat",
            genotype_file=r"TutorialData\mdp_genotype.hmp.txt",
            phenotype_file=r"TutorialData\mdp_traits.txt",
            output_dir="outputs",
            memory="8g"
    ):

        self.tassel_path = tassel_path
        self.genotype_file = genotype_file
        self.phenotype_file = phenotype_file
        self.output_dir = output_dir
        self.memory = memory

        os.makedirs(self.output_dir, exist_ok=True)

    # =====================================================
    # 执行命令
    # =====================================================
    def run_command(self, command):

        print("\n========== Running ==========")
        print(" ".join(command))
        print("================================\n")

        result = subprocess.run(command)

        if result.returncode != 0:
            raise RuntimeError("TASSEL command failed.")

    # =====================================================
    # 1. SNP过滤
    # =====================================================
    def filter_snp(
            self,
            maf=0.05,
            min_count=150
    ):

        export_prefix = os.path.join(
            self.output_dir,
            "filtered"
        )

        command = [

            self.tassel_path,

            f"-Xmx{self.memory}",

            "-fork1",

            "-h",
            self.genotype_file,

            "-filterAlign",

            "-filterAlignMinFreq",
            str(maf),

            "-filterAlignMinCount",
            str(min_count),

            "-export",
            export_prefix,

            "-exportType",
            "Hapmap"
        ]

        self.run_command(command)

        # TASSEL 自动生成:
        # filtered1.hmp.txt
        filtered_file = export_prefix + "1.hmp.txt"

        return filtered_file

    # =====================================================
    # 2. PCA分析
    # =====================================================
    def run_pca(self, filtered_file):

        export_prefix = os.path.join(
            self.output_dir,
            "pca"
        )

        command = [

            self.tassel_path,

            f"-Xmx{self.memory}",

            "-fork1",

            "-h",
            filtered_file,

            "-PrincipalComponentsPlugin",

            "-endPlugin",

            "-export",
            export_prefix
        ]

        self.run_command(command)

        pca_file = export_prefix + "1.txt"

        return pca_file

    # =====================================================
    # 3. Kinship矩阵
    # =====================================================
    def calculate_kinship(self, filtered_file):

        export_prefix = os.path.join(
            self.output_dir,
            "kinship"
        )

        command = [

            self.tassel_path,

            f"-Xmx{self.memory}",

            "-fork1",

            "-h",
            filtered_file,

            "-KinshipPlugin",

            "-endPlugin",

            "-export",
            export_prefix
        ]

        self.run_command(command)

        kinship_file = export_prefix + ".txt"

        return kinship_file

    # =====================================================
    # 4. MLM GWAS
    # =====================================================
    def run_mlm(
            self,
            filtered_file,
            kinship_file,
            pca_file=None
    ):

        output_file = os.path.join(
            self.output_dir,
            "mlm_result.txt"
        )

        # -------------------------------------------------
        # 不使用 PCA
        # -------------------------------------------------
        if pca_file is None:

            command = [

                self.tassel_path,

                f"-Xmx{self.memory}",

                "-fork1",
                "-h", filtered_file,

                "-fork2",
                "-r", self.phenotype_file,

                "-fork3",
                "-k", kinship_file,

                "-combine4",

                "-input1",
                "-input2",

                "-intersect",

                "-combine5",

                "-input4",
                "-input3",

                "-mlm",

                "-mlmVarCompEst",
                "P3D",

                "-mlmCompressionLevel",
                "Optimum",

                "-mlmOutputFile",
                output_file
            ]

        # -------------------------------------------------
        # 使用 PCA
        # -------------------------------------------------
        else:

            command = [

                self.tassel_path,

                f"-Xmx{self.memory}",

                "-fork1",
                "-h", filtered_file,

                "-fork2",
                "-r", self.phenotype_file,

                "-fork3",
                "-q", pca_file,

                "-fork4",
                "-k", kinship_file,

                "-combine5",

                "-input1",
                "-input2",
                "-input3",

                "-intersect",

                "-combine6",

                "-input5",
                "-input4",

                "-mlm",

                "-mlmVarCompEst",
                "P3D",

                "-mlmCompressionLevel",
                "Optimum",

                "-mlmOutputFile",
                output_file
            ]

        self.run_command(command)

        return output_file

    # =====================================================
    # 5. 自动完整流程
    # =====================================================
    def run_pipeline(self):

        print("\n==============================")
        print("Step 1: SNP Filtering")
        print("==============================")

        filtered_file = self.filter_snp()

        print("\n==============================")
        print("Step 2: PCA Analysis")
        print("==============================")

        pca_file = self.run_pca(filtered_file)

        print("\n==============================")
        print("Step 3: Kinship Matrix")
        print("==============================")

        kinship_file = self.calculate_kinship(filtered_file)

        print("\n==============================")
        print("Step 4: MLM GWAS")
        print("==============================")

        result_file = self.run_mlm(
            filtered_file,
            kinship_file,
            pca_file
        )

        print("\n==============================")
        print("GWAS Finished!")
        print("==============================")

        print("Filtered File :", filtered_file)
        print("PCA File      :", pca_file)
        print("Kinship File  :", kinship_file)
        print("Result File   :", result_file)


# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":

    pipeline = TasselGWASPipeline(

        tassel_path=r"..\TASSEL5\run_pipeline.bat",

        genotype_file=r"TutorialData\mdp_genotype.hmp.txt",

        phenotype_file=r"TutorialData\mdp_traits.txt",

        output_dir="outputs",

        memory="8g"
    )

    pipeline.run_pipeline()