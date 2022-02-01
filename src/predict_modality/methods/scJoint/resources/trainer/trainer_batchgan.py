"""
trainer of batchgan architecture
works better on gex2adt subtask
"""
import os
import logging
import numpy as np
import anndata as ad
from scipy.sparse import csc_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from opts import DATASET
from utils.metric import rmse
from utils.loss import L1regularization
from utils.dataloader import BatchSeqDataset
from modules.model_ae import BatchRemovalGAN


class TrainProcess:
    """the training process for batch gan arch"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu_ids[0]}") if args.gpu_ids else torch.device("cpu")

        mod1_idf = np.load(args.idf_path) if args.tfidf != 0 else None
        self.trainset = BatchSeqDataset(
            DATASET[args.mode]["train_mod1"],
            DATASET[args.mode]["train_mod2"],
            mod1_idx_path=args.mod1_idx_path,
            tfidf=args.tfidf,
            mod1_idf=mod1_idf,
            batch_list=args.train_batch,
        )

        # do train/test split of training dataset in phase 2
        if args.phase in ["phase1", "phase1v2"]:
            self.testset = BatchSeqDataset(
                DATASET[args.mode]["test_mod1"],
                DATASET[args.mode]["test_mod2"],
                mod1_idx_path=args.mod1_idx_path,
                tfidf=args.tfidf,
                mod1_idf=mod1_idf,
                batch_list=args.test_batch,
            )

        else:
            self.testset = BatchSeqDataset(
                DATASET[args.mode]["train_mod1"],
                DATASET[args.mode]["train_mod2"],
                mod1_idx_path=args.mod1_idx_path,
                tfidf=args.tfidf,
                mod1_idf=mod1_idf,
                batch_list=args.test_batch,
            )

            self.testset2 = BatchSeqDataset(
                DATASET[args.mode]["test_mod1"],
                DATASET[args.mode]["test_mod2"],
                mod1_idx_path=args.mod1_idx_path,
                tfidf=args.tfidf,
                mod1_idf=mod1_idf,
                batch_list=[],
            )

        logging.info(f"TRAIN_NUM: {len(self.trainset):5d}")
        logging.info(f"VAL_NUM : {len(self.testset):5d}")

        self.train_loader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.testset, batch_size=args.batch_size, shuffle=False)

        if args.phase == "phase2":
            self.test_loader2 = DataLoader(self.testset2, batch_size=args.batch_size, shuffle=False)
            logging.info(f"TEST_NUM : {len(self.testset2):5d}")
        else:
            self.test_loader2 = None

        self.model = (
            BatchRemovalGAN(
                input_dim=args.mod1_dim,
                out_dim=args.mod2_dim,
                feat_dim=args.emb_dim,
                hidden_dim=args.hid_dim,
                dropout=args.dropout,
            )
            .to(self.device)
            .float()
        )

        logging.info(self.model)

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.l1reg_loss = L1regularization(weight_decay=0.1)
        self.adv_loss = nn.CrossEntropyLoss()
        self.eval_best = float("inf")

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=5e-4,
        )

        self.optimizer_g = optim.Adam(self.model.parameters(), lr=args.lr * 2e-3, betas=[0.5, 0.999])
        self.optimizer_d = optim.Adam(self.model.parameters(), lr=args.lr * 2e-3, betas=[0.5, 0.999])

    def adjust_learning_rate(self, optimizer, epoch):
        """learning rate adjustment method"""
        lr = self.args.lr * (0.5 ** ((epoch - 0) // self.args.lr_decay_epoch))
        if (epoch - 0) % self.args.lr_decay_epoch == 0:
            print("LR is set to {}".format(lr))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def load_checkpoint(self):
        """load pre-trained model checkpoint"""
        if self.args.checkpoint is not None:
            if os.path.isfile(self.args.checkpoint):
                logging.info(f"loading checkpoint: {self.args.checkpoint}")
                checkpoint = torch.load(self.args.checkpoint)
                self.model.load_state_dict(checkpoint)
            else:
                logging.info(f"no resume checkpoint found at {self.args.checkpoint}")

    def train_epoch(self, epoch):
        """training process of each epoch"""
        self.model.train()
        self.adjust_learning_rate(self.optimizer, epoch)

        total_loss = 0.0
        total_rec_loss = 0.0
        print(f"Epoch {epoch+1:2d} / {self.args.epoch}")

        for batch_idx, (mod1_seq, mod2_seq, b_label) in enumerate(self.train_loader):

            mod1_seq = mod1_seq.to(self.device).float()
            mod2_seq = mod2_seq.to(self.device).float()
            b_label = b_label.to(self.device).long()

            # Reconstruction phase
            mod2_rec, _ = self.model(mod1_seq)
            rec_loss = self.mse_loss(mod2_rec, mod2_seq)
            l1reg_loss = self.l1reg_loss(self.model) * self.args.reg_loss_weight
            loss = rec_loss + l1reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # G phase
            _, cls_prob = self.model(mod1_seq)
            g_loss = -self.adv_loss(cls_prob, b_label) * 0.1  # maximize class loss

            self.optimizer_g.zero_grad()
            g_loss.backward()
            self.optimizer_g.step()

            # D phase
            _, cls_prob = self.model(mod1_seq)
            d_loss = self.adv_loss(cls_prob, b_label) * 0.1  # minimize class loss

            self.optimizer_d.zero_grad()
            d_loss.backward()
            self.optimizer_d.step()

            total_loss += loss.item()
            total_rec_loss += rec_loss.item()

            print(
                f"Epoch {epoch+1:2d} [{batch_idx+1:2d} /{len(self.train_loader):2d}] | "
                + f"Total: {total_loss / (batch_idx + 1):.4f} | "
                + f"Rec L2: {rec_loss.item():.4f} | "
                + f"L1 Reg: {l1reg_loss.item():.4f} | "
                + f"G: {g_loss.item() :.5f} | "
                + f"D: {d_loss.item() :.5f} | "
            )

        train_rmse = np.sqrt(total_rec_loss / len(self.train_loader))
        test_rmse = self.eval_epoch()
        (self.eval_best, save_best) = (test_rmse, True) if test_rmse < self.eval_best else (self.eval_best, False)

        logging.info(
            f"Epoch {epoch+1:3d} / {self.args.epoch} | Train RMSE: {train_rmse:.4f}"
            + f"| Eval RMSE: {test_rmse:.4f} | Eval best: {self.eval_best:.4f}"
        )

        # save checkpoint
        if not self.args.dryrun:
            filename = f"{DATASET[self.args.mode]['weight_dir']}/model_{self.args.exp_name}.pt"
            print(f"saving weight to {filename} ...")
            torch.save(self.model.state_dict(), filename)

            # best weight
            if save_best and epoch > self.args.save_best_from:
                filename = f"{DATASET[self.args.mode]['weight_dir']}/model_best_{self.args.exp_name}.pt"
                print(f"saving best weight to {filename} ...")
                torch.save(self.model.state_dict(), filename)

    def eval_epoch(self):
        """eval process of each epoch"""
        self.model.eval()

        total_rec_loss = 0.0
        for _, (mod1_seq, mod2_seq, _) in enumerate(self.test_loader):
            mod1_seq = mod1_seq.to(self.device).float()
            mod2_seq = mod2_seq.to(self.device).float()
            mod2_rec, _ = self.model(mod1_seq)

            rec_loss = self.mse_loss(mod2_rec, mod2_seq)
            total_rec_loss += rec_loss.item()
        test_rmse = np.sqrt(total_rec_loss / len(self.test_loader))

        return test_rmse

    def run(self):
        """run the whole training process"""
        self.load_checkpoint()
        print("start training ...")
        for epoch in range(self.args.epoch):
            self.train_epoch(epoch)

    def eval(self):
        """eval the trained model on train / test set"""
        print("start eval...")
        self.model.eval()

        logging.info(f"Mode: {self.args.mode}")

        DATAS = {
            "train": self.train_loader,
            "eval": self.test_loader,
            "test": self.test_loader2,
        }
        eval_set = ["train", "eval", "test"] if self.args.phase == "phase2" else ["train", "eval"]
        if self.args.phase == "phase1" and self.args.mode != "atac2gex":
            use_numpy = False
        else:
            use_numpy = True

        for mode in eval_set:
            if use_numpy:
                mod2_pred_matrix = np.zeros((1, self.args.mod2_dim))
                mod2_gt_matrix = np.zeros((1, self.args.mod2_dim))
            else:
                mod2_pred_matrix = []
                mod2_gt_matrix = []

            # enumerate with current set
            for _, (mod1_seq, mod2_seq, _) in enumerate(DATAS[mode]):
                mod1_seq = mod1_seq.to(self.device).float()
                mod2_seq = mod2_seq.to(self.device).float()
                mod2_pred, _ = self.model(mod1_seq)

                if use_numpy:
                    mod2_pred = mod2_pred.data.cpu().numpy()
                    mod2_pred_matrix = np.vstack((mod2_pred_matrix, mod2_pred))

                    mod2_gt = mod2_seq.data.cpu().numpy()
                    mod2_gt_matrix = np.vstack((mod2_gt_matrix, mod2_gt))

                else:
                    mod2_pred_matrix.append(mod2_pred)
                    mod2_gt_matrix.append(mod2_gt)

            if use_numpy:
                mod2_pred_matrix = mod2_pred_matrix[
                    1:,
                ]
                mod2_gt_matrix = mod2_gt_matrix[
                    1:,
                ]
            else:
                mod2_pred_matrix = torch.cat(mod2_pred_matrix).detach().cpu().numpy()
                mod2_gt_matrix = torch.cat(mod2_gt_matrix).detach().cpu().numpy()

            mod2_pred_matrix = csc_matrix(mod2_pred_matrix)
            mod2_gt_matrix = csc_matrix(mod2_gt_matrix)

            # calculate rmse
            rmse_pred = rmse(mod2_gt_matrix, mod2_pred_matrix)
            logging.info(f"{mode.upper()} RMSE: {rmse_pred:5f}")
