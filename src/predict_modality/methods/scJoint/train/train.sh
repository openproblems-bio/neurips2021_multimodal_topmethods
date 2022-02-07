#!/bin/bash

## VIASH START
meta_resources_dir="src/predict_modality/methods/scJoint/"
meta_method_id="scjoint"

par_data_dir="output/datasets/predict_modality/"
par_pretrain_dir="output/pretrain/predict_modality/"
par_output_pretrain="${par_pretrain_dir}${meta_method_id}/"
## VIASH END

# create
[ -d "$par_output_pretrain" ] || mkdir -p "$par_output_pretrain" && echo "$par_output_pretrain"

# preprocess files
echo ""
echo "######################################################################"
echo "##                      Checking preprocess files                   ##"
echo "######################################################################"
idf_list=( "$par_output_pretrain/gex2adt_train.output_pretrain/mod1_idf.npy" 
           "$par_output_pretrain/adt2gex_train.output_pretrain/mod1_idf.npy" 
           "$par_output_pretrain/gex2atac_train.output_pretrain/mod1_idf.npy" 
           "$par_output_pretrain/atac2gex_train.output_pretrain/mod1_idf.npy" )
for file in ${idf_list[@]}; do
    if [ -f "${file}" ]; then
        preprocess=false
        break
    else
        preprocess=true
    fi
done

if [ "$preprocess" = true ]; then
    echo "Generating preprocess files"
    python3 "$meta_resources_dir/resources/preprocess/save_idf_matrix.py" \
        -d "$par_data_dir" -o "$par_output_pretrain"
else
    echo "Preprocess files exist"
fi

# CITE GEX2ADT
echo ""
echo "######################################################################"
echo "##                      Pretraining GEX2ADT                         ##"
echo "######################################################################"
pretrain_dir="gex2adt_train.output_pretrain"

# train set a
echo "Pretrain (1a)"
if [ -f "$par_output_pretrain/$pretrain_dir/model_nn_gex2adt_pretrain1a.pt" ]; then
    echo "nn: model_nn_gex2adt_pretrain1a.pt exist"
else
    echo "training nn: model_nn_gex2adt_pretrain1a.pt"
    python3 "$meta_resources_dir/resources/train.py" \
        --mode gex2adt --arch nn \
        --train_batch s1d1 s1d3 s2d1 s2d4 s2d5 s3d1 s3d6 \
        --test_batch s1d2 s3d7 \
        --data_dir "$par_data_dir" \
        --output_dir "$par_output_pretrain" \
        -dp 0.2 -e 300 --lr_decay_epoch 40 --reg_loss_weight 1 \
        --gpu_ids 1 --name pretrain1a
fi

# train set b
echo "Pretrain (2b)"
if [ -f "$par_output_pretrain/$pretrain_dir/model_nn_gex2adt_pretrain2b.pt" ]; then
    echo "nn: model_nn_gex2adt_pretrain2b.pt exist"
else
    echo "training nn: model_nn_gex2adt_pretrain2b.pt"
    python3 "$meta_resources_dir/resources/train.py" \
        --mode gex2adt --arch nn \
        --train_batch s1d2 s1d3 s2d1 s2d5 s3d1 s3d6 s3d7 \
        --test_batch s1d1 s2d4 \
        --data_dir "$par_data_dir" \
        --output_dir "$par_output_pretrain" \
        -dp 0.2 -e 300 --lr_decay_epoch 40 --reg_loss_weight 1 \
        --gpu_ids 1 --name pretrain2b
fi

# train set a 
echo "Pretrain (3a)"
if [ -f "$par_output_pretrain/$pretrain_dir/model_nn_gex2adt_tfidfconcat_pretrain3a.pt" ]; then
    echo "tfidfconcat: model_nn_gex2adt_tfidfconcat_pretrain3a.pt exist"
else
    echo "training tfidfconcat: model_nn_gex2adt_tfidfconcat_pretrain3a.pt"
    python3 "$meta_resources_dir/resources/train.py" \
        --mode gex2adt --arch nn \
        --train_batch s1d1 s1d3 s2d1 s2d4 s2d5 s3d1 s3d6 \
        --test_batch s1d2 s3d7 \
        --data_dir "$par_data_dir" \
        --output_dir "$par_output_pretrain" \
        --tfidf 2 --idf_path "$par_output_pretrain/$pretrain_dir/mod1_idf.npy" \
        -dp 0.2 -e 300 --lr_decay_epoch 40 --reg_loss_weight 1 \
        --gpu_ids 1 --name pretrain3a
fi

# train set b
echo "Pretrain (4b)"
if [ -f "$par_output_pretrain/$pretrain_dir/model_nn_gex2adt_tfidfconcat_pretrain4b.pt" ]; then
    echo "tfidfconcat: model_nn_gex2adt_tfidfconcat_pretrain4b.pt exist"
else
    echo "training tfidfconcat: model_nn_gex2adt_tfidfconcat_pretrain4b.pt"
    python3 "$meta_resources_dir/resources/train.py" \
        --mode gex2adt --arch nn \
        --train_batch s1d2 s1d3 s2d1 s2d5 s3d1 s3d6 s3d7 \
        --test_batch s1d1 s2d4 \
        --data_dir "$par_data_dir" \
        --output_dir "$par_output_pretrain" \
        --tfidf 2 --idf_path "$par_output_pretrain/$pretrain_dir/mod1_idf.npy" \
        -dp 0.2 -e 300 --lr_decay_epoch 40 --reg_loss_weight 1 \
        --gpu_ids 1 --name pretrain4b
fi

# train set c
echo "Pretrain (5c)"
if [ -f "$par_output_pretrain/$pretrain_dir/model_nn_gex2adt_pretrain5c.pt" ]; then
    echo "nn: model_nn_gex2adt_pretrain5c.pt exist"
else
    echo "training nn: model_nn_gex2adt_pretrain5c.pt"
    python3 "$meta_resources_dir/resources/train.py" \
        --mode gex2adt --arch nn \
        --train_batch s1d1 s1d2 s1d3 s2d1 s2d4 s3d1 s3d7 \
        --test_batch s2d5 s3d6 \
        --data_dir "$par_data_dir" \
        --output_dir "$par_output_pretrain" \
        -dp 0.2 -e 300 --lr_decay_epoch 40 --reg_loss_weight 1 \
        --gpu_ids 1 --name pretrain5c
fi

# train set d
echo "Pretrain (6d)"
if [ -f "$par_output_pretrain/$pretrain_dir/model_nn_gex2adt_pretrain6d.pt" ]; then
    echo "nn: model_nn_gex2adt_pretrain6d.pt exist"
else
    echo "training nn: model_nn_gex2adt_pretrain6d.pt"
    python3 "$meta_resources_dir/resources/train.py" \
        --mode gex2adt --arch nn \
        --train_batch s1d1 s1d2 s2d1 s2d4 s2d5 s3d6 s3d7 \
        --test_batch s1d3 s3d1 \
        --data_dir "$par_data_dir" \
        --output_dir "$par_output_pretrain" \
        -dp 0.2 -e 300 --lr_decay_epoch 40 --reg_loss_weight 1 \
        --gpu_ids 1 --name pretrain6d
fi

# train set c 
echo "Pretrain (7c)"
if [ -f "$par_output_pretrain/$pretrain_dir/model_nn_gex2adt_tfidfconcat_pretrain7c.pt" ]; then
    echo "tfidfconcat: model_nn_gex2adt_tfidfconcat_pretrain7c.pt exist"
else
    echo "training tfidfconcat: model_nn_gex2adt_tfidfconcat_pretrain7c.pt"
    python3 "$meta_resources_dir/resources/train.py" \
        --mode gex2adt --arch nn \
        --train_batch s1d1 s1d2 s1d3 s2d1 s2d4 s3d1 s3d7 \
        --test_batch s2d5 s3d6 \
        --data_dir "$par_data_dir" \
        --output_dir "$par_output_pretrain" \
        --tfidf 2 --idf_path "$par_output_pretrain/$pretrain_dir/mod1_idf.npy" \
        -dp 0.2 -e 300 --lr_decay_epoch 40 --reg_loss_weight 1 \
        --gpu_ids 1 --name pretrain7c
fi

# train set d
echo "Pretrain (8d)"
if [ -f "$par_output_pretrain/$pretrain_dir/model_nn_gex2adt_tfidfconcat_pretrain8d.pt" ]; then
    echo "tfidfconcat: model_nn_gex2adt_tfidfconcat_pretrain8d.pt exist"
else
    echo "training tfidfconcat: model_nn_gex2adt_tfidfconcat_pretrain8d.pt"
    python3 "$meta_resources_dir/resources/train.py" \
        --mode gex2adt --arch nn \
        --train_batch s1d1 s1d2 s2d1 s2d4 s2d5 s3d6 s3d7 \
        --test_batch s1d3 s3d1 \
        --data_dir "$par_data_dir" \
        --output_dir "$par_output_pretrain" \
        --tfidf 2 --idf_path "$par_output_pretrain/$pretrain_dir/mod1_idf.npy" \
        -dp 0.2 -e 300 --lr_decay_epoch 40 --reg_loss_weight 1 \
        --gpu_ids 1 --name pretrain8d
fi

# CITE ADT2GEX
echo ""
echo "######################################################################"
echo "##                      Pretraining ADT2GEX                         ##"
echo "######################################################################"
pretrain_dir="adt2gex_train.output_pretrain"
# train set d
echo "Pretrain (1d)"
if [ -f "$par_output_pretrain/$pretrain_dir/model_cycle_adt2gex_tfidfconcat_pretrain1d.pt" ]; then
    echo "tfidfconcat: model_cycle_adt2gex_tfidfconcat_pretrain1d.pt exist"
else
    echo "training tfidfconcat: model_cycle_adt2gex_tfidfconcat_pretrain1d.pt"
    python3 "$meta_resources_dir/resources/train.py" \
        --mode adt2gex --arch cycle \
        --train_batch s1d1 s1d2 s2d1 s2d4 s2d5 s3d6 s3d7 \
        --test_batch s1d3 s3d1 \
        --data_dir "$par_data_dir" \
        --output_dir "$par_output_pretrain" \
        --tfidf 2 --idf_path "$par_output_pretrain/$pretrain_dir/mod1_idf.npy" \
        -e 400 --lr_decay_epoch 80 \
        --gpu_ids 1 --name pretrain1d
fi

# train set d
echo "Pretrain (2d)"
if [ -f "$par_output_pretrain/$pretrain_dir/model_cycle_adt2gex_pretrain2d.pt" ]; then
    echo "cycle: model_cycle_adt2gex_pretrain2d.pt exist"
else
    echo "training cycle: model_cycle_adt2gex_pretrain2d.pt"
    python3 "$meta_resources_dir/resources/train.py" \
        --mode adt2gex --arch cycle \
        --train_batch s1d1 s1d2 s2d1 s2d4 s2d5 s3d6 s3d7 \
        --test_batch s1d3 s3d1 \
        --data_dir "$par_data_dir" \
        --output_dir "$par_output_pretrain" \
        -e 400 --lr_decay_epoch 80 \
        --gpu_ids 1 --name pretrain2d
fi

# MULTIOME GEX2ATAC
echo ""
echo "######################################################################"
echo "##                      Pretraining GEX2ATAC                        ##"
echo "######################################################################"
pretrain_dir="gex2atac_train.output_pretrain"

echo "Pretrain (1)"
if [ -f "$par_output_pretrain/$pretrain_dir/model_cycle_gex2atac_tfidfconcat_pretrain1.pt" ]; then
    echo "tfidfconcat: model_cycle_gex2atac_tfidfconcat_pretrain1.pt exist"
else
    echo "training tfidfconcat: model_cycle_gex2atac_tfidfconcat_pretrain1.pt"
    python3 "$meta_resources_dir/resources/train.py" \
        --mode gex2atac --arch cycle \
        --train_batch s1d1 s1d3 s2d1 s2d4 s2d5 s3d10 s3d3 s3d6 \
        --test_batch s1d2 s3d7 \
        --data_dir "$par_data_dir" \
        --output_dir "$par_output_pretrain" \
        --tfidf 2 --idf_path "$par_output_pretrain/$pretrain_dir/mod1_idf.npy" \
        -e 400 --lr_decay_epoch 80 \
        --gpu_ids 1 --name pretrain1
fi

# MULTIOME ATAC2GEX
echo ""
echo "######################################################################"
echo "##                      Pretraining ATAC2GEX                        ##"
echo "######################################################################"
pretrain_dir="atac2gex_train.output_pretrain"

# train set b
echo "Pretrain (1b)"
if [ -f "$par_output_pretrain/$pretrain_dir/model_cycle_atac2gex_ga_pretrain1b.pt" ]; then
    echo "ga: model_cycle_atac2gex_ga_pretrain1b.pt exist"
else
    echo "training ga: model_cycle_atac2gex_ga_pretrain1b.pt"
    python3 "$meta_resources_dir/resources/train.py" \
        --mode atac2gex --arch cycle --gene_activity \
        --train_batch s1d1 s1d2 s1d3 s2d1 s2d5 s3d3 s3d7 \
        --test_batch s2d4 s3d6 s3d10 \
        --data_dir "$par_data_dir" \
        --output_dir "$par_output_pretrain" \
        -e 400 --lr_decay_epoch 40 -lr 0.2 \
        --gpu_ids 1 --name pretrain1b
fi

# train set b
echo "Pretrain (2b)"
if [ -f "$par_output_pretrain/$pretrain_dir/model_cycle_atac2gex_ga_pretrain2b.pt" ]; then
    echo "ga: model_cycle_atac2gex_ga_pretrain2b.pt exist"
else
    echo "training ga: model_cycle_atac2gex_ga_pretrain2b.pt"
    python3 "$meta_resources_dir/resources/train.py" \
        --mode atac2gex --arch cycle --gene_activity \
        --train_batch s1d1 s1d2 s1d3 s2d1 s2d5 s3d3 s3d7 \
        --test_batch s2d4 s3d6 s3d10 \
        --data_dir "$par_data_dir" \
        --output_dir "$par_output_pretrain" \
        -e 400 --lr_decay_epoch 40 -lr 0.2 \
        --gpu_ids 1 --name pretrain2b
fi
