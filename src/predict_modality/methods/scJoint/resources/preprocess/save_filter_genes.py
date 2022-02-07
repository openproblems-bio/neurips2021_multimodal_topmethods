"""
saved filtered features using the scanpy package

current dataset stats:
ATAC2GEX_phase2
mod 1 (43492, 116490)
{pct: 5, min_cells: 2174, feature_num: 15196}
{pct: 2, min_cells: 869, feature_num: 34072}

GEX2ADT_phase2
mod 1 (67175, 13953)
{pct: 5, min_cells: 3358, feature_num: 7065}
{pct: 2, min_cells: 1343, feature_num: 10058}
"""
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
    default=["atac2gex", "gex2adt"],
    help="modes for generating idf matrix",
)

parser.add_argument("-pct", "--pct", type=int, default=5, help="modes for generating idf matrix")

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

        train_mod1 = ad.read_h5ad(train_mod1_pth)
        test_mod1 = ad.read_h5ad(test_mod1_pth)

        # concat train/test sets
        X_raw = ad.concat(
            {"train": train_mod1, "test": test_mod1},
            axis=0,
            join="outer",
            label="group",
            fill_value=0,
            index_unique="-",
        )
        min_cells = int(X_raw.X.shape[0] * 0.01 * args.pct)
        print(f"min cells: {min_cells}")
        print(X_raw.shape)

        # collect n pct genes
        sc.pp.filter_genes(X_raw, min_cells=min_cells)

        train_npct = X_raw[: train_mod1.X.shape[0], :]
        train_npct = ad.AnnData(
            X=train_npct.X,
            obs=train_npct.obs,
            var=pd.DataFrame({"feature_types": train_mod1.var["feature_types"][X_raw.var_names]}),
            uns=train_npct.uns,
            layers=train_npct.layers,
        )

        test_npct = X_raw[train_mod1.X.shape[0] :, :]
        test_npct = ad.AnnData(
            X=test_npct.X,
            obs=test_npct.obs,
            var=pd.DataFrame({"feature_types": test_mod1.var["feature_types"][X_raw.var_names]}),
            uns=test_npct.uns,
            layers=test_npct.layers,
        )
        print(train_npct)
        print(test_npct)

        # save npct indexs
        mod1_vars = np.array(train_mod1.var_names)
        mod1_npct_idx = [
            int(np.where(mod1_vars == np.array(X_raw.var_names[i]))[0])
            for i in range(np.array(X_raw.var_names).shape[0])
        ]
        file_path = f"{mode[4]}"
        os.makedirs(file_path, exist_ok=True)

        with open(f"{file_path}/index_{args.pct}pct.txt", "w", encoding="utf8") as index_file:
            index_file.write(f"index num: {len(mod1_npct_idx)}\n")
            for ind in mod1_npct_idx:
                index_file.write(str(ind) + "\n")

        print(f"finish saving {file_path}/index_{args.pct}pct.txt")
