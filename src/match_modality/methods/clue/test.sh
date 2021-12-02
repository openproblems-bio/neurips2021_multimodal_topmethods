#!/bin/bash

DATASET_PATH=output/datasets_phase2/phase2/match_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset
PRETRAIN_PATH=output/pretrain/clue/openproblems_bmmc_cite_phase1v2_rna.clue_train.output_pretrain/
PRED_PATH=output/predictions/phase2/match_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.

target/docker/match_modality_methods/train/clue_train \
  --input_train_mod1 ${DATASET_PATH}.output_train_mod1.h5ad \
  --input_train_mod2 ${DATASET_PATH}.output_train_mod2.h5ad \
  --input_train_sol ${DATASET_PATH}.output_train_sol.h5ad \
  --output_pretrain $PRETRAIN_PATH

target/docker/match_modality_methods/train/clue_run \
  --input_train_mod1 ${DATASET_PATH}.output_train_mod1.h5ad \
  --input_train_mod2 ${DATASET_PATH}.output_train_mod2.h5ad \
  --input_train_sol ${DATASET_PATH}.output_train_sol.h5ad \
  --input_test_mod1 ${DATASET_PATH}.output_test_mod1.h5ad \
  --input_test_mod2 ${DATASET_PATH}.output_test_mod2.h5ad \
  --input_pretrain $PRETRAIN_PATH \
  --output ${PRED_PATH}.clue_run.output.h5ad