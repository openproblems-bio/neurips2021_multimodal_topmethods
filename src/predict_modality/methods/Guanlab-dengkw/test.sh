#!/bin/bash

export PIPELINE_REPO="openproblems-bio/neurips2021_multimodal_viash"
export NXF_VER=21.04.1
export PIPELINE_VERSION=1.4.0
method_id=guanlab_dengkw_pm
task_id=predict_modality

# CITE GEX2ADT
dataset_id=openproblems_bmmc_cite_phase2_rna
dataset_path=output/datasets/$task_id/$dataset_id/$dataset_id.censor_dataset
pred_path=output/predictions/$task_id/$dataset_id/$dataset_id

target/docker/${task_id}_methods/${method_id}/${method_id} \
  --input_train_mod1 ${dataset_path}.output_train_mod1.h5ad \
  --input_train_mod2 ${dataset_path}.output_train_mod2.h5ad \
  --input_test_mod1 ${dataset_path}.output_test_mod1.h5ad \
  --output ${pred_path}.${method_id}.output.h5ad

# CITE ADT2GEX
dataset_id=openproblems_bmmc_cite_phase2_mod2
dataset_path=output/datasets/$task_id/$dataset_id/$dataset_id.censor_dataset
pred_path=output/predictions/$task_id/$dataset_id/$dataset_id

target/docker/${task_id}_methods/${method_id}/${method_id} \
  --input_train_mod1 ${dataset_path}.output_train_mod1.h5ad \
  --input_train_mod2 ${dataset_path}.output_train_mod2.h5ad \
  --input_test_mod1 ${dataset_path}.output_test_mod1.h5ad \
  --input_pretrain ${pretrain_path} \
  --output ${pred_path}.${method_id}.output.h5ad

# MULTIOME GEX2ATAC
dataset_id=openproblems_bmmc_multiome_phase2_rna
dataset_path=output/datasets/$task_id/$dataset_id/$dataset_id.censor_dataset
pred_path=output/predictions/$task_id/$dataset_id/$dataset_id

target/docker/${task_id}_methods/${method_id}/${method_id} \
  --input_train_mod1 ${dataset_path}.output_train_mod1.h5ad \
  --input_train_mod2 ${dataset_path}.output_train_mod2.h5ad \
  --input_test_mod1 ${dataset_path}.output_test_mod1.h5ad \
  --input_pretrain ${pretrain_path} \
  --output ${pred_path}.${method_id}.output.h5ad

# MULTIOME ATAC2GEX
dataset_id=openproblems_bmmc_multiome_phase2_mod2
dataset_path=output/datasets/$task_id/$dataset_id/$dataset_id.censor_dataset
pred_path=output/predictions/$task_id/$dataset_id/$dataset_id

target/docker/${task_id}_methods/${method_id}/${method_id} \
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
