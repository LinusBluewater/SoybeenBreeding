import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests


class GWASSNPSelector:

    def __init__(
        self,
        gwas_file,
        top_n=1000,
        flank_size=50000,
        correction="fdr_bh"
    ):
        self.gwas_file = gwas_file
        self.top_n = top_n
        self.flank_size = flank_size
        self.correction = correction

        self.df = None
        self.significant_snps = None

    # =========================
    # load
    # =========================
    def load_data(self):
        self.df = pd.read_csv(self.gwas_file, sep=None, engine='python')
        print(f"Loaded {len(self.df)} GWAS records")

    # =========================
    # clean
    # =========================
    def preprocess(self):

        self.df = self.df.dropna(subset=["Trait", "Marker", "p", "Chr", "Pos"])
        self.df = self.df[self.df["Marker"] != "None"]

        for col in ["p", "Pos", "MarkerR2"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self.df = self.df.dropna()
        self.df = self.df[(self.df["p"] > 0) & (self.df["p"] <= 1)]

    # =========================
    # correction
    # =========================
    def multiple_testing(self):

        if self.correction == "none":
            self.df["adj_p"] = self.df["p"]
        else:
            self.df["adj_p"] = multipletests(
                self.df["p"],
                method=self.correction
            )[1]

    # =========================
    # selection
    # =========================
    def select_top_snps(self):

        self.df = self.df.sort_values("adj_p")

        sig = self.df.head(self.top_n).copy()

        sig["RegionStart"] = (sig["Pos"] - self.flank_size).clip(lower=0)
        sig["RegionEnd"] = sig["Pos"] + self.flank_size

        self.significant_snps = sig

        print(f"Selected top {len(sig)} SNPs")

    # =========================
    # export for CNN
    # =========================
    def export_snp_list(self, output_file):

        self.significant_snps.to_csv(
            output_file,
            sep="\t",
            index=False
        )

        print(f"Saved SNP list to {output_file}")

        # 👉 返回给CNN用
        return self.significant_snps

    # =========================
    # full pipeline
    # =========================
    def run(self, output_file):

        self.load_data()
        self.preprocess()
        self.multiple_testing()
        self.select_top_snps()

        return self.export_snp_list(output_file)