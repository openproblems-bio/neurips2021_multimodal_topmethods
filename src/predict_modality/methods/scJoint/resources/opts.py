""" opt file for the model """
import argparse


def model_opts(parser):
    """model opts"""
    # model / data setting
    parser.add_argument(
        "--data_dir",
        type=str,
        default="output/datasets/predict_modality",
        help="Path to dataset directory",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/pretrain/predict_modality/scjoint",
        help="Path to output directory",
    )

    parser.add_argument(
        "--phase",
        default="phase2",
        type=str,
        choices=["phase1", "phase1v2", "phase2"],
        help="Dataset phase",
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "gex2atac",
            "gex2adt",
            "adt2gex",
            "atac2gex",
        ],
        help="Desired training mode",
    )

    parser.add_argument(
        "--train",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Training or evaluating the model",
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="nn",
        choices=["nn", "cycle", "batchgan"],
        help="Desired training architecture",
    )

    parser.add_argument(
        "--train_batch",
        type=str,
        nargs="*",
        default=[],
        help="Desired training batch \
            v1: train = ['s1d1', 's2d1', 's2d4', 's3d6'], v1 test  = ['s1d2'] \
            v2 train = ['s1d1', 's1d3', 's2d1', 's2d4', 's2d5', 's3d1', 's3d3', 's3d6', 's3d10'] \
            v2 test  = ['s1d2', 's3d7'] \
            p3 train = [s1: d1, d2, d3; s2: d1, d4, d5; s3: d1, d3, d6, d7, d10]",
    )
    parser.add_argument("--test_batch", type=str, nargs="*", default=[], help="Desired testing batch")

    # optimization
    parser.add_argument("--epoch", "-e", type=int, default=200)
    parser.add_argument("--batch_size", "-bs", type=int, default=2048)
    parser.add_argument("--lr", "-lr", type=float, default=0.1)
    parser.add_argument("--lr_decay_epoch", type=int, default=40)
    parser.add_argument("--momentum", type=float, default=0.9)

    # model architecture
    parser.add_argument("--dropout", "-dp", type=float, default=0.2)
    parser.add_argument("--hid_dim", type=int, default=1000)
    parser.add_argument("--emb_dim", type=int, default=128)

    # loss functions
    parser.add_argument("--reg_loss_weight", type=float, default=0)
    parser.add_argument("--rec_loss_weight", type=float, default=10)

    # data preprocessing
    parser.add_argument(
        "--norm",
        action="store_true",
        help="True for normalize mod1 input data batch-wise",
    )
    parser.add_argument(
        "--gene_activity",
        action="store_true",
        help="True for use gene activity feature in mod1 input, \
            Can be apply only on atac2gex* mode",
    )
    parser.add_argument(
        "--selection",
        action="store_true",
        help="True for using the selected feature index",
    )
    parser.add_argument(
        "--mod1_idx_path",
        type=str,
        default=None,
        help="The path to mod1 index path (.np), required when selection=True",
    )
    parser.add_argument(
        "--tfidf",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="The tfidf mode. \
        0: raw data input \
        1: tfidf input \
        2: concat [raw, tfidf] feature \
        3: concat [gene activity, tfidf] feature",
    )
    parser.add_argument(
        "--idf_path",
        type=str,
        default=None,
        help="The path to pre-calculated idf matrix, required if tfidf != 0",
    )

    # save/load model
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="True: saves weights, logs, runs (tensorboard) during training, \
            False: saves runs (tensorboard) only during training",
    )
    parser.add_argument("--save_best_from", "-best", type=int, default=50)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained model checkpoint",
    )

    # others
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0, use -1 for CPU")


parser = argparse.ArgumentParser(add_help=False)
model_opts(parser)
args = parser.parse_known_args()[0]

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

ADT2GEX = {
    "train_mod1": f"{ADT2GEX_PTH}.output_train_mod1.h5ad",
    "train_mod2": f"{ADT2GEX_PTH}.output_train_mod2.h5ad",
    "test_mod1": f"{ADT2GEX_PTH}.output_test_mod1.h5ad",
    "test_mod2": f"{ADT2GEX_PTH}.output_test_mod2.h5ad",
    "weight_dir": f"{args.output_dir}/adt2gex_train.output_pretrain",
}

GEX2ADT = {
    "train_mod1": f"{GEX2ADT_PTH}.output_train_mod1.h5ad",
    "train_mod2": f"{GEX2ADT_PTH}.output_train_mod2.h5ad",
    "test_mod1": f"{GEX2ADT_PTH}.output_test_mod1.h5ad",
    "test_mod2": f"{GEX2ADT_PTH}.output_test_mod2.h5ad",
    "weight_dir": f"{args.output_dir}/gex2adt_train.output_pretrain",
}

ATAC2GEX = {
    "train_mod1": f"{ATAC2GEX_PTH}.output_train_mod1.h5ad",
    "train_mod2": f"{ATAC2GEX_PTH}.output_train_mod2.h5ad",
    "test_mod1": f"{ATAC2GEX_PTH}.output_test_mod1.h5ad",
    "test_mod2": f"{ATAC2GEX_PTH}.output_test_mod2.h5ad",
    "weight_dir": f"{args.output_dir}/atac2gex_train.output_pretrain",
}

GEX2ATAC = {
    "train_mod1": f"{GEX2ATAC_PTH}.output_train_mod1.h5ad",
    "train_mod2": f"{GEX2ATAC_PTH}.output_train_mod2.h5ad",
    "test_mod1": f"{GEX2ATAC_PTH}.output_test_mod1.h5ad",
    "test_mod2": f"{GEX2ATAC_PTH}.output_test_mod2.h5ad",
    "weight_dir": f"{args.output_dir}/gex2atac_train.output_pretrain",
}

# datasets
DATASET = {
    "atac2gex": ATAC2GEX,
    "adt2gex": ADT2GEX,
    "gex2adt": GEX2ADT,
    "gex2atac": GEX2ATAC,
}
