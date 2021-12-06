#!/bin/bash

export PIPELINE_REPO="openproblems-bio/neurips2021_multimodal_viash"
export NXF_VER=21.04.1
export PIPELINE_VERSION=1.4.0
method_id=submission_170825
task_id=joint_embedding

# CITE
dataset_id=openproblems_bmmc_cite_phase2
dataset_path=output/datasets/$task_id/$dataset_id/$dataset_id.censor_dataset
pred_path=output/predictions/$task_id/$dataset_id/$dataset_id

target/docker/${task_id}_methods/run/${method_id} \
  --input_mod1 ${dataset_path}.output_mod1.h5ad \
  --input_mod2 ${dataset_path}.output_mod2.h5ad \
  --output ${pred_path}.${method_id}.output.h5ad

# MULTIOME
dataset_id=openproblems_bmmc_multiome_phase2
dataset_path=output/datasets/$task_id/$dataset_id/$dataset_id.censor_dataset
pred_path=output/predictions/$task_id/$dataset_id/$dataset_id

target/docker/${task_id}_methods/run/${method_id} \
  --input_mod1 ${dataset_path}.output_mod1.h5ad \
  --input_mod2 ${dataset_path}.output_mod2.h5ad \
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