""" this function save idf matrixs from the dataset """
import os
import argparse
import numpy as np
import anndata as ad

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
    default="phase2",
    type=str,
    choices=["phase1", "phase1v2", "phase2"],
    help="dataset phase",
)

parser.add_argument(
    "-m",
    "--mode",
    type=str,
    nargs="*",
    default=["adt2gex", "gex2adt", "atac2gex", "gex2atac"],
    help="modes for generating idf matrix",
)
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


def idf_matrix(x_raw):
    """returns idf matrix"""
    x_idf = np.zeros_like(x_raw).astype(np.single)
    x_idf[x_raw > 0] = 1
    idf = np.log(x_raw.shape[0] / (np.sum(x_idf, axis=0, keepdims=True) + 1))
    return idf


if __name__ == "__main__":
    # desired data path
    DATAPTH = [MODES[i] for i in args.mode]
    for (i, mode) in enumerate(DATAPTH):
        print(f"MODE [{i + 1} / {len(DATAPTH)}]: {args.mode[i]}")

        train_mod1_pth = mode[0]
        train_mod1 = ad.read_h5ad(train_mod1_pth)

        x_raw_matrix = train_mod1.layers["counts"].toarray().astype(np.float16)
        print(f"train data shape: {x_raw_matrix.shape}")

        x_idf_matrix = idf_matrix(x_raw_matrix)
        print(f"idf matrix shape: {x_idf_matrix.shape}")

        file_path = f"{mode[4]}"
        print(f"output dir: {file_path}")
        os.makedirs(file_path, exist_ok=True)

        np.save(f"{file_path}/mod1_idf.npy", x_idf_matrix)
        print(f"finish saving {file_path}/mod1_idf.npy")
