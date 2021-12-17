#!/bin/bash

export PIPELINE_REPO="openproblems-bio/neurips2021_multimodal_viash"
export NXF_VER=21.04.1
export PIPELINE_VERSION=1.4.0
method_id=submission_171129
task_id=predict_modality

# GENERATE PRETRAIN
pretrain_path=output/pretrain/$task_id/$method_id/pretrain.${method_id}_train.output_pretrain/

target/docker/${task_id}_methods/train/${method_id}_train \
  --data_dir output/datasets/$task_id \
  --output_pretrain ${pretrain_path}

# CITE GEX2ADT
dataset_id=openproblems_bmmc_cite_phase2_rna
dataset_path=output/datasets/$task_id/$dataset_id/$dataset_id.censor_dataset
pred_path=output/predictions/$task_id/$dataset_id/$dataset_id

target/docker/${task_id}_methods/run/${method_id} \
  --input_train_mod1 ${dataset_path}.output_train_mod1.h5ad \
  --input_train_mod2 ${dataset_path}.output_train_mod2.h5ad \
  --input_test_mod1 ${dataset_path}.output_test_mod1.h5ad \
  --input_pretrain ${pretrain_path} \
  --output ${pred_path}.${method_id}.output.h5ad

# CITE ADT2GEX
dataset_id=openproblems_bmmc_cite_phase2_mod2
dataset_path=output/datasets/$task_id/$dataset_id/$dataset_id.censor_dataset
pred_path=output/predictions/$task_id/$dataset_id/$dataset_id

target/docker/${task_id}_methods/run/${method_id} \
  --input_train_mod1 ${dataset_path}.output_train_mod1.h5ad \
  --input_train_mod2 ${dataset_path}.output_train_mod2.h5ad \
  --input_test_mod1 ${dataset_path}.output_test_mod1.h5ad \
  --input_pretrain ${pretrain_path} \
  --output ${pred_path}.${method_id}.output.h5ad


# MULTIOME GEX2ATAC
dataset_id=openproblems_bmmc_multiome_phase2_rna
dataset_path=output/datasets/$task_id/$dataset_id/$dataset_id.censor_dataset
pretrain_path=output/pretrain/$task_id/$method_id/$dataset_id.${method_id}_train.output_pretrain/
pred_path=output/predictions/$task_id/$dataset_id/$dataset_id

target/docker/${task_id}_methods/train/${method_id}_train \
  --input_train_mod1 ${dataset_path}.output_train_mod1.h5ad \
  --input_train_mod2 ${dataset_path}.output_train_mod2.h5ad \
  --output_pretrain ${pretrain_path}

target/docker/${task_id}_methods/run/${method_id} \
  --input_train_mod1 ${dataset_path}.output_train_mod1.h5ad \
  --input_train_mod2 ${dataset_path}.output_train_mod2.h5ad \
  --input_test_mod1 ${dataset_path}.output_test_mod1.h5ad \
  --input_pretrain ${pretrain_path} \
  --output ${pred_path}.${method_id}.output.h5ad

# MULTIOME ATAC2GEX
dataset_id=openproblems_bmmc_multiome_phase2_mod2
dataset_path=output/datasets/$task_id/$dataset_id/$dataset_id.censor_dataset
pred_path=output/predictions/$task_id/$dataset_id/$dataset_id

target/docker/${task_id}_methods/run/${method_id} \
  --input_train_mod1 ${dataset_path}.output_train_mod1.h5ad \
  --input_train_mod2 ${dataset_path}.output_train_mod2.h5ad \
  --input_test_mod1 ${dataset_path}.output_test_mod1.h5ad \
  --input_pretrain ${pretrain_path} \
  --output ${pred_path}.${method_id}.output.h5ad

# RUN EVALUATION
bin/nextflow run "$PIPELINE_REPO" \
  -r "$PIPELINE_VERSION" \
  -main-script "src/$task_id/workflows/evaluate_submission/main.nf" \
  --solutionDir "output/datasets/$task_id" \
  --predictions "output/predictions/$task_id/**.${method_id}.output.h5ad" \
  --publishDir "output/evaluation/$task_id/$method_id/" \
  -latest \
  -resume \
  -c "src/resources/nextflow_moremem.config"

cat "output/evaluation/$task_id/$method_id/output.final_scores.output_json.json"