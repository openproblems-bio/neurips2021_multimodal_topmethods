""" save highly variable using scanpy package """
import os
import argparse
import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    default="output/datasets/predict_modality",
    help="path to dataset directory",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default="output/pretrain/predict_modality/scjoint",
    help="path to output directory",
)
parser.add_argument(
    "-p",
    "--phase",
    type=str,
    default="phase2",
    choices=["phase1", "phase1v2", "phase2"],
    help="dataset phase",
)

parser.add_argument(
    "-m",
    "--mode",
    nargs="*",
    type=str,
    default=["atac2gex"],
    help="modes for generating idf matrix",
)

parser.add_argument("-n", "--n_top", type=int, default=10000, help="returns n top highly variable genes")
args = parser.parse_args()

# datset path
ADT2GEX_ID = f"openproblems_bmmc_cite_{args.phase}_mod2"
GEX2ADT_ID = f"openproblems_bmmc_cite_{args.phase}_rna"
ATAC2GEX_ID = f"openproblems_bmmc_multiome_{args.phase}_mod2"
GEX2ATAC_ID = f"openproblems_bmmc_multiome_{args.phase}_rna"

# path to different modes
ADT2GEX_PTH = f"{args.data_dir}/{ADT2GEX_ID}/{ADT2GEX_ID}.censor_dataset"
GEX2ADT_PTH = f"{args.data_dir}/{GEX2ADT_ID}/{GEX2ADT_ID}.censor_dataset"
ATAC2GEX_PTH = f"{args.data_dir}/{ATAC2GEX_ID}/{ATAC2GEX_ID}.censor_dataset"
GEX2ATAC_PTH = f"{args.data_dir}/{GEX2ATAC_ID}/{GEX2ATAC_ID}.censor_dataset"

ADT2GEX = [
    f"{ADT2GEX_PTH}.output_train_mod1.h5ad",
    f"{ADT2GEX_PTH}.output_train_mod2.h5ad",
    f"{ADT2GEX_PTH}.output_test_mod1.h5ad",
    f"{ADT2GEX_PTH}.output_test_mod2.h5ad",
    f"{args.output_dir}/adt2gex_train.output_pretrain",
]

GEX2ADT = [
    f"{GEX2ADT_PTH}.output_train_mod1.h5ad",
    f"{GEX2ADT_PTH}.output_train_mod2.h5ad",
    f"{GEX2ADT_PTH}.output_test_mod1.h5ad",
    f"{GEX2ADT_PTH}.output_test_mod2.h5ad",
    f"{args.output_dir}/gex2adt_train.output_pretrain",
]

ATAC2GEX = [
    f"{ATAC2GEX_PTH}.output_train_mod1.h5ad",
    f"{ATAC2GEX_PTH}.output_train_mod2.h5ad",
    f"{ATAC2GEX_PTH}.output_test_mod1.h5ad",
    f"{ATAC2GEX_PTH}.output_test_mod2.h5ad",
    f"{args.output_dir}/atac2gex_train.output_pretrain",
]

GEX2ATAC = [
    f"{GEX2ATAC_PTH}.output_train_mod1.h5ad",
    f"{GEX2ATAC_PTH}.output_train_mod2.h5ad",
    f"{GEX2ATAC_PTH}.output_test_mod1.h5ad",
    f"{GEX2ATAC_PTH}.output_test_mod2.h5ad",
    f"{args.output_dir}/gex2atac_train.output_pretrain",
]

MODES = {"adt2gex": ADT2GEX, "gex2adt": GEX2ADT, "atac2gex": ATAC2GEX, "gex2atac": GEX2ATAC}


if __name__ == "__main__":
    # desired data path
    DATAPTH = [MODES[i] for i in args.mode]

    for (i, mode) in enumerate(DATAPTH):
        print(f"MODE [{i + 1} / {len(DATAPTH)}]: {args.mode[i]}")
        train_mod1_pth = mode[0]
        test_mod1_pth = mode[2]
        train_mod1 = sc.read_h5ad(train_mod1_pth)
        test_mod1 = sc.read_h5ad(test_mod1_pth)

        # concat train/test sets
        X_raw = sc.concat(
            {"train": train_mod1, "test": test_mod1},
            axis=0,
            join="outer",
            label="group",
            fill_value=0,
            index_unique="-",
        )
        print(X_raw.shape)

        # collect highly variable genes
        sc.pp.highly_variable_genes(X_raw, n_top_genes=args.n_top)
        X_raw = X_raw[:, X_raw.var.highly_variable]

        train_highly = X_raw[: train_mod1.X.shape[0], :]
        train_highly = ad.AnnData(
            X=train_highly.X,
            obs=train_highly.obs,
            var=pd.DataFrame({"feature_types": train_mod1.var["feature_types"][X_raw.var_names]}),
            uns=train_highly.uns,
            layers=train_highly.layers,
        )

        test_highly = X_raw[train_mod1.X.shape[0] :, :]
        test_highly = ad.AnnData(
            X=test_highly.X,
            obs=test_highly.obs,
            var=pd.DataFrame({"feature_types": test_mod1.var["feature_types"][X_raw.var_names]}),
            uns=test_highly.uns,
            layers=test_highly.layers,
        )
        print(train_highly)
        print(test_highly)

        # save highly variable indexs
        mod1_vars = np.array(train_mod1.var_names)
        mod1_highly_idx = [
            int(np.where(mod1_vars == np.array(X_raw.var_names[i]))[0])
            for i in range(np.array(X_raw.var_names).shape[0])
        ]

        file_path = f"{mode[4]}"
        os.makedirs(file_path, exist_ok=True)

        with open(f"{file_path}/index_highly{args.n_top}.txt", "w", encoding="utf8") as index_file:
            index_file.write(f"index num: {len(mod1_highly_idx)}\n")
            for ind in mod1_highly_idx:
                index_file.write(str(ind) + "\n")

        print(f"finish saving {file_path}/index_highly{args.n_top}.txt")
