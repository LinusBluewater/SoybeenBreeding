import yaml
import os
from src.gwas import TasselGWAS


cfg = yaml.safe_load(open("configs/config.yaml"))

os.makedirs(cfg["paths"]["gwas_out"], exist_ok=True)


gwas = TasselGWAS(
    tassel_path=r"..\TASSEL5\run_pipeline.bat",
    memory="8g"
)

out = cfg["paths"]["gwas_out"]

filtered = gwas.filter_snp(
    genotype_file=cfg["paths"]["geno"],
    output_prefix=os.path.join(out, "filtered")
)

pca = gwas.run_pca(filtered, os.path.join(out, "pca"))

kinship = gwas.run_kinship(filtered, os.path.join(out, "kinship"))

gwas.run_mlm(
    filtered,
    cfg["paths"]["pheno"],
    kinship,
    os.path.join(out, "mlm_result.txt"),
    pca_file=pca
)

print("GWAS DONE")