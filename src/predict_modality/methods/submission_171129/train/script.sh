#!/bin/bash

## VIASH START
meta_resources_dir="src/predict_modality/methods/submission_171129/train/"
meta_functionality_name="submission_171129"

par_data_dir="output/datasets/predict_modality/"
par_output_pretrain="output/pretrain/predict_modality/$meta_functionality_name/pretrain.${meta_functionality_name}_train.output_pretrain/"
## VIASH END

# create 
[ -d "$par_output_pretrain" ] || mkdir -p "$par_output_pretrain"

echo "Copying gmt files to pretrain dir"
cp "$meta_resources_dir/h.all.v7.4.entrez.gmt" "$meta_resources_dir/h.all.v7.4.symbols.gmt" "$par_output_pretrain"


echo "Generating extra files"
python "$meta_resources_dir/generate_extra_files.py" \
    -d "$par_data_dir" -ef "$par_output_pretrain"

echo "Generating 'bf_alpha_conv4_mean_fullbatch_8000_phase2_inductive_gex2atac.pkl'"
python "$meta_resources_dir/hetero_arg_version_v5.py" 'bf_alpha_conv4_mean_fullbatch_8000_phase2_inductive_gex2atac' \
    -pww 'cos' -res 'res_cat' -inres -pwagg 'alpha' -pwalpha=0.5 -bs=60000 -nm 'group' -ac 'gelu' \
    -em=2 -ro=1 -conv=4 -agg 'mean' -sf -lr=1e-2 -wd=1e-5 -hid=48 -edd=0.4 -mdd=0.2 -e=8000 -sb -i 'normal' \
    -t 'openproblems_bmmc_multiome_phase2_rna' \
    -m "$par_output_pretrain" -r "$par_output_pretrain" -l "$par_output_pretrain" -d "$par_data_dir" -ef "$par_output_pretrain"

echo "Generating 'bf_alpha_conv4_mean_fullbatch_8000_phase2_inductive_gex2atac_2.pkl'"
python "$meta_resources_dir/hetero_arg_version_v5.py" 'bf_alpha_conv4_mean_fullbatch_8000_phase2_inductive_gex2atac_2' \
    -pww 'cos' -res 'res_cat' -inres -pwagg 'alpha' -pwalpha=0.5 -bs=60000 -nm 'group' -ac 'gelu' \
    -em=2 -ro=1 -conv=4 -agg 'mean' -sf -lr=1e-2 -wd=1e-5 -hid=48 -edd=0.4 -mdd=0.2 -e=10000 -sb -i 'normal' \
    -t 'openproblems_bmmc_multiome_phase2_rna' \
    -m "$par_output_pretrain" -r "$par_output_pretrain" -l "$par_output_pretrain" -d "$par_data_dir" -ef "$par_output_pretrain"

echo "Generating 'bf_alpha_conv4_mean_fullbatch_8000_phase2_inductive_gex2atac_3.pkl'"
python "$meta_resources_dir/hetero_arg_version_v5.py" 'bf_alpha_conv4_mean_fullbatch_8000_phase2_inductive_gex2atac_3' \
    -pww 'cos' -res 'res_cat' -inres -pwagg 'alpha' -pwalpha=0.5 -bs=60000 -nm 'group' -ac 'gelu' \
    -em=2 -ro=1 -conv=4 -agg 'mean' -sf -lr=1e-2 -wd=1e-5 -hid=48 -edd=0.4 -mdd=0.2 -e=10000 -sb -i 'normal' \
    -t 'openproblems_bmmc_multiome_phase2_rna' \
    -m "$par_output_pretrain" -r "$par_output_pretrain" -l "$par_output_pretrain" -d "$par_data_dir" -ef "$par_output_pretrain"

echo "Generating 'bf_alpha_conv4_mean_fullbatch_10000_phase2_inductive_gex2atac.pkl'"
python "$meta_resources_dir/hetero_arg_version_v5.py" 'bf_alpha_conv4_mean_fullbatch_10000_phase2_inductive_gex2atac' \
    -pww 'cos' -res 'res_cat' -inres -pwagg 'alpha' -pwalpha=0.5 -bs=60000 -nm 'group' -ac 'gelu' \
    -em=2 -ro=1 -conv=4 -agg 'mean' -sf -lr=1e-2 -wd=1e-5 -hid=48 -edd=0.4 -mdd=0.2 -e=10000 -sb -i 'normal' \
    -t 'openproblems_bmmc_multiome_phase2_rna' \
    -m "$par_output_pretrain" -r "$par_output_pretrain" -l "$par_output_pretrain" -d "$par_data_dir" -ef "$par_output_pretrain"

echo "Generating 'bf_alpha_conv4_mean_fullbatch_10000_phase2_inductive_gex2adt_2.pkl'"
python "$meta_resources_dir/hetero_arg_version_v5.py" 'bf_alpha_conv4_mean_fullbatch_10000_phase2_inductive_gex2adt_2' \
    -pww 'cos' -res 'res_cat' -inres -pwagg 'alpha' -pwalpha=0.5 -bs=60000 -nm 'group' -ac 'gelu' \
    -em=2 -ro=1 -conv=4 -agg 'mean' -sf -lr=1e-2 -wd=1e-5 -hid=48 -edd=0.4 -mdd=0.2 -e=10000 -sb -i 'normal' -bas \
    -t 'openproblems_bmmc_cite_phase2_rna' \
    -m "$par_output_pretrain" -r "$par_output_pretrain" -l "$par_output_pretrain" -d "$par_data_dir" -ef "$par_output_pretrain"

echo "Generating 'bf_alpha_conv4_mean_fullbatch_12000_phase2_inductive_gex2adt_sep_2.pkl'"
python "$meta_resources_dir/hetero_arg_version_v5.py" 'bf_alpha_conv4_mean_fullbatch_12000_phase2_inductive_gex2adt_sep_2' \
    -pww 'cos' -res 'res_cat' -inres -pwagg 'alpha' -pwalpha=0.5 -bs=60000 -nm 'group' -ac 'gelu' \
    -em=2 -ro=1 -conv=4 -agg 'mean' -sf -lr=1e-2 -wd=1e-5 -hid=48 -edd=0.4 -mdd=0.2 -e=12000 -sb -i 'normal' -bas \
    -t 'openproblems_bmmc_cite_phase2_rna' \
    -m "$par_output_pretrain" -r "$par_output_pretrain" -l "$par_output_pretrain" -d "$par_data_dir" -ef "$par_output_pretrain"

echo "Generating 'bf_alpha_conv4_mean_fullbatch_15000_phase2_inductive.pkl'"
python "$meta_resources_dir/hetero_arg_version_v5.py" 'bf_alpha_conv4_mean_fullbatch_15000_phase2_inductive' \
    -pww 'cos' -res 'res_cat' -inres -pwagg 'alpha' -pwalpha=0.5 -bs=60000 -nm 'group' -ac 'gelu' \
    -em=2 -ro=1 -conv=4 -agg 'mean' -sf -lr=1e-2 -wd=1e-5 -hid=48 -edd=0.4 -mdd=0.2 -e=12000 -sb -i 'normal' \
    -t 'openproblems_bmmc_cite_phase2_rna' \
    -m "$par_output_pretrain" -r "$par_output_pretrain" -l "$par_output_pretrain" -d "$par_data_dir" -ef "$par_output_pretrain"

echo "Generating 'f_alpha_conv4_mean_fullbatch_12000_phase2_inductive_batch_speration.pkl'"
python "$meta_resources_dir/hetero_arg_version_v5.py" 'f_alpha_conv4_mean_fullbatch_12000_phase2_inductive_batch_speration' \
    -pww 'cos' -res 'res_cat' -inres -pwagg 'alpha' -pwalpha=0.5 -bs=60000 -nm 'group' -ac 'gelu' \
    -em=2 -ro=1 -conv=4 -agg 'mean' -sf -lr=1e-2 -wd=1e-5 -hid=48 -edd=0.4 -mdd=0.2 -e=12000 -sb -i 'normal' -bas \
    -t 'openproblems_bmmc_cite_phase2_rna' \
    -m "$par_output_pretrain" -r "$par_output_pretrain" -l "$par_output_pretrain" -d "$par_data_dir" -ef "$par_output_pretrain"
