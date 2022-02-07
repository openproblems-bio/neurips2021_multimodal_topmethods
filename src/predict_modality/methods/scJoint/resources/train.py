""" main training process """
import os
import logging
import argparse
from datetime import datetime

from trainer.trainer_nn import TrainProcess as TrainProcess_NN
from trainer.trainer_cycle import TrainProcess as TrainProcess_Cycle
from trainer.trainer_batchgan import TrainProcess as TrainProcess_BATCHGAN

from opts import DATASET, model_opts
from utils.dataloader import get_data_dim

if __name__ == "__main__":
    # config parser
    parser = argparse.ArgumentParser(add_help=False)
    model_opts(parser)
    args = parser.parse_known_args()[0]

    # exp name for train log, weights, model
    if args.train == "train":
        TIME_NOW = datetime.now().strftime("%b%d-%H-%M")
        exp_name = f"{args.arch}_{args.mode}"
        if args.selection:
            assert args.mod1_idx_path is not None, "need to specified --mod1_idx_path"
            SELECT_NUM = args.mod1_idx_path.split("/")[-1].replace(".txt", "").replace("index_", "")
            exp_name += f"_select{SELECT_NUM}"

        if args.tfidf != 0:
            assert args.idf_path is not None, "need to specified --idf_path"
            assert not args.gene_activity, "support either ga or tfidf != 0"
            if args.tfidf == 1:
                exp_name += f"_tfidf"
            elif args.tfidf == 2:
                exp_name += f"_tfidfconcat"
            elif args.tfidf == 3:
                exp_name += f"_tfidfconcatga"
                assert args.mode == "atac2gex" and args.phase in [
                    "phase1v2",
                    "phase2",
                ], "gene activity mode support only atac2gex mode (p1v2 or p2)"
        elif args.gene_activity:
            exp_name += f"_ga"
            assert args.mode == "atac2gex" and args.phase in [
                "phase1v2",
                "phase2",
            ], "gene activity mode support only atac2gex mode (p1v2 or p2)"
        if args.norm:
            exp_name += f"_norm"
        if args.dropout != 0.2:
            exp_name += f"_dropout{args.dropout}"
        if args.name != "":
            exp_name += f"_{args.name}"
        else:
            exp_name += f"_{TIME_NOW}"

    # exp name for eval log file
    elif args.train == "eval":
        assert args.checkpoint is not None, "need to specified --checkpoint"
        exp_name = args.checkpoint.split("/")[-1].replace(".pt", "")
        exp_name += f"_{args.phase}"

    # loggings and logs
    if args.dryrun:
        handlers = [logging.StreamHandler()]
    else:
        os.makedirs(f"{args.output_dir}/logs/", exist_ok=True)
        os.makedirs(f"{DATASET[args.mode]['weight_dir']}", exist_ok=True)
        handlers = [
            logging.FileHandler(f"{args.output_dir}/logs/{args.train}_{exp_name}.log", mode="w"),
            logging.StreamHandler(),
        ]

    logging.basicConfig(level=logging.DEBUG, format="%(message)s", handlers=handlers)

    # load data
    MOD1_DIM, MOD2_DIM = get_data_dim(DATASET[args.mode], args)

    parser.add_argument("--mod1_dim", default=MOD1_DIM)
    parser.add_argument("--mod2_dim", default=MOD2_DIM)
    parser.add_argument("--exp_name", default=exp_name)
    args = parser.parse_args()

    logging.info("\nArgument:")
    for arg, value in vars(args).items():
        logging.info(f"{arg:20s}: {value}")
    logging.info("\n")

    # trainer
    if args.arch == "nn":
        trainer = TrainProcess_NN(args)
    elif args.arch == "cycle":
        trainer = TrainProcess_Cycle(args)
    elif args.arch == "batchgan":
        trainer = TrainProcess_BATCHGAN(args)

    if args.train == "train":
        trainer.run()
        trainer.eval()

    elif args.train == "eval":
        trainer.load_checkpoint()
        trainer.eval()
