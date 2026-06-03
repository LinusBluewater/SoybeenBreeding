import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests


class GWASSNPSelector:

    def __init__(
            self,
            gwas_file,
            p_threshold=1000,
            flank_size=50000,
            correction="fdr_bh"
    ):

        """
        Parameters
        ----------
        gwas_file : str
            GWAS结果文件

        p_threshold : float
            显著性阈值

        flank_size : int
            SNP上下游区域大小

        correction : str
            多重检验校正方式:
            - bonferroni
            - fdr_bh
            - none
        """

        self.gwas_file = gwas_file
        self.p_threshold = p_threshold
        self.flank_size = flank_size
        self.correction = correction

        self.df = None
        self.significant_snps = None

    def load_data(self):

        # 自动识别分隔符
        self.df = pd.read_csv(
            self.gwas_file,
            sep=None,
            engine='python'
        )

        print(f"Loaded {len(self.df)} GWAS records")

    def preprocess(self):

        self.df = self.df.dropna(
            subset=["Trait", "Marker", "p", "Chr", "Pos"]
        )

        self.df = self.df[
            self.df["Marker"] != "None"
            ]

        numeric_cols = [
            "p",
            "Pos",
            "MarkerR2"
        ]

        for col in numeric_cols:
            self.df[col] = pd.to_numeric(
                self.df[col],
                errors="coerce"
            )

        self.df = self.df.dropna()

        # 删除非法p值
        self.df = self.df[
            (self.df["p"] > 0) &
            (self.df["p"] <= 1)
            ]

    def multiple_testing_correction(self):

        if self.correction == "none":
            self.df["adjusted_p"] = self.df["p"]

        else:

            adjusted = multipletests(
                self.df["p"],
                method=self.correction
            )[1]

            self.df["adjusted_p"] = adjusted

    def select_significant_snps(self):

        # if self.correction == "none":
        #
        #     sig = self.df[
        #         self.df["p"] < self.p_threshold
        #     ]
        #
        # else:
        #
        #     sig = self.df[
        #         self.df["adjusted_p"] < self.p_threshold
        #     ]

        sig = self.df.sort_values(
            by="adjusted_p"
        ).head(self.p_threshold)

        # 按Trait分组
        sig = sig.sort_values(
            by=["Trait", "adjusted_p"]
        )

        # 添加区域信息
        sig["RegionStart"] = (
                sig["Pos"] - self.flank_size
        ).clip(lower=0)

        sig["RegionEnd"] = (
                sig["Pos"] + self.flank_size
        )

        self.significant_snps = sig

        print(
            f"Found {len(sig)} significant SNPs"
        )

    def summarize_traits(self):

        summary = (
            self.significant_snps
            .groupby("Trait")
            .size()
            .reset_index(name="SignificantSNPCount")
        )

        print("\nTrait Summary:")
        print(summary)

        return summary

    def export_results(self, output_file):

        columns = [
            "Trait",
            "Marker",
            "Chr",
            "Pos",
            "p",
            "adjusted_p",
            "MarkerR2",
            "add_effect",
            "dom_effect",
            "RegionStart",
            "RegionEnd"
        ]

        self.significant_snps[
            columns
        ].to_csv(
            output_file,
            sep='\t',
            index=False
        )

        print(f"\nSaved to: {output_file}")

    def run(self, output_file):

        self.load_data()

        self.preprocess()

        self.multiple_testing_correction()

        self.select_significant_snps()

        self.summarize_traits()

        self.export_results(output_file)


if __name__ == "__main__":

    selector = GWASSNPSelector(
        gwas_file=r"outputs\mlm_result.txt_filtered1_+_mdp_traits_+_pca1_stats.txt",

        # 提取前多少个p值
        p_threshold=1000,

        # 上下游50kb
        flank_size=50000,

        # 推荐
        correction="fdr_bh"
    )

    selector.run(
        output_file="significant_snps.tsv"
    )