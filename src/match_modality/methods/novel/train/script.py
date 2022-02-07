import sys
from sklearn.model_selection import LeaveOneGroupOut

import numpy as np
import anndata as ad
import torch
from torch import nn
import torch.nn.functional as F

import pandas as pd
import shutil
import pickle

from catalyst import dl, utils
import catalyst
import os

from sklearn.model_selection import LeaveOneGroupOut

## VIASH START
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=""

dataset_path = "output/datasets/match_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_"
pretrain_path = "output/pretrain/match_modality/clue/openproblems_bmmc_cite_phase2_rna.clue_train.output_pretrain/"

par = {
    'input_train_mod1': f'{dataset_path}train_mod1.h5ad',
    'input_train_mod2': f'{dataset_path}train_mod2.h5ad',
    'input_train_sol': f'{dataset_path}train_sol.h5ad',
    'input_test_mod1': f'{dataset_path}test_mod1.h5ad',
    'input_test_mod2': f'{dataset_path}test_mod2.h5ad',
    'input_test_sol': f'{dataset_path}test_sol.h5ad',
    'output_pretrain': pretrain_path
}
meta = {
    'resources_dir': '.',
    'functionality_name': '169594'
}
## VIASH END


sys.path.append(meta['resources_dir'])
from data import get_dataloaders, ModalityMatchingDataset
from models import Modality_CLIP, Encoder, symmetric_npair_loss
from catalyst_tools import scRNARunner, CustomMetric
from preprocessing import lsiTransformer

os.makedirs(par['output_pretrain'], exist_ok=True)

print("Start train")

input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
sol_train = ad.read_h5ad(par['input_train_sol'])

mod1 = input_train_mod1.var['feature_types'][0]
mod2 = input_train_mod2.var['feature_types'][0]

input_train_mod2 = input_train_mod2[sol_train.to_df().values.argmax(1)]
    
if(mod1 == 'ADT' or mod2 == 'ADT'):
    import config_ADT2GEX as config
    
    logo = LeaveOneGroupOut()
    groups = sol_train.obs.batch
    logo.get_n_splits(input_train_mod2, groups=groups)
    
    for fold_number, (train_indexes, test_indexes) in enumerate(logo.split(input_train_mod2, groups=groups)):
        trial_dump_folder = os.path.join(par['output_pretrain'], str(fold_number))
        os.makedirs(trial_dump_folder, exist_ok=True)
        
        lsi_transformer_gex = lsiTransformer(n_components=config.N_LSI_COMPONENTS_GEX, drop_first=True)
        
        if(mod1 == 'ADT'):
            gex_train = lsi_transformer_gex.fit_transform(input_train_mod2[train_indexes])
            gex_test = lsi_transformer_gex.transform(input_train_mod2[test_indexes])
            adt_train = input_train_mod1[train_indexes].to_df()
            adt_test = input_train_mod1[test_indexes].to_df()
            
            dataset_train = ModalityMatchingDataset(adt_train, gex_train)
            dataset_test = ModalityMatchingDataset(adt_test, gex_test)
            
            dataloader_train = torch.utils.data.DataLoader(dataset_train, config.BATCH_SIZE, shuffle = True)
            dataloader_test = torch.utils.data.DataLoader(dataset_test, 2048, shuffle = False)
            
                
            model = Modality_CLIP(
                Encoder=Encoder, 
                layers_dims = (
                  config.LAYERS_DIM_FIRST, 
                  config.LAYERS_DIM_GEX
                ),
                dropout_rates = (
                  config.DROPOUT_RATES_FIRST, 
                  config.DROPOUT_RATES_GEX
                ),
                dim_mod1 = 134 if mod1 == 'ADT' else config.N_LSI_COMPONENTS_FIRST,
                dim_mod2 = config.N_LSI_COMPONENTS_GEX, 
                output_dim = config.EMBEDDING_DIM,
                T = config.LOG_T,
                swap_rate_1 = 0.,
                swap_rate_2 = 0.)
        
            optimizer = torch.optim.Adam(model.parameters(), config.LR, weight_decay=config.weight_decay)
            loaders = {
                "train": dataloader_train,
                "valid": dataloader_test,
            }
            runner = scRNARunner()
            
            runner.train(
                model=model,
                optimizer=optimizer,
                loaders=loaders,
                num_epochs=config.N_EPOCHS,
                callbacks=[
                    dl.OptimizerCallback(metric_key='loss'),
                    dl.CheckpointCallback(
                        logdir = trial_dump_folder,
                        loader_key='valid',
                        metric_key='avg_acc',
                        minimize=False,
                        use_runner_logdir=False,
                        save_n_best=1
                    ),
                    dl.EarlyStoppingCallback(
                        patience=150,
                        loader_key='valid',
                        metric_key='avg_acc',
                        minimize=False,
                        min_delta=1e-5),
                    dl.LoaderMetricCallback(
                        metric=CustomMetric(),
                        input_key=['embeddings_first', 'embeddings_second', 'temperature'],
                        target_key=['embeddings_second']
                    ),
                ],
                verbose=True
            )
            
            with open(trial_dump_folder + '/lsi_transformer.pickle', 'wb') as f:
                pickle.dump(lsi_transformer_gex, f)
    
    
else:
    import config_ATAC2GEX as config
    test_indexes = sol_train.obs.batch == 's1d1'
    train_indexes = sol_train.obs.batch != 's1d1'
    
    lsi_transformer_atac = lsiTransformer(n_components=config.N_LSI_COMPONENTS_FIRST, drop_first=True)
    lsi_transformer_gex = lsiTransformer(n_components=config.N_LSI_COMPONENTS_GEX, drop_first=True)
    
    if(mod1 == 'ATAC'):
        atac_train = lsi_transformer_atac.fit_transform(input_train_mod1[train_indexes])
        atac_test = lsi_transformer_atac.transform(input_train_mod1[test_indexes])
        
        gex_train = lsi_transformer_gex.fit_transform(input_train_mod2[train_indexes])
        gex_test = lsi_transformer_gex.transform(input_train_mod2[test_indexes])
        
    dataset_train = ModalityMatchingDataset(atac_train, gex_train)
    dataset_test = ModalityMatchingDataset(atac_test, gex_test)
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, config.BATCH_SIZE, shuffle = True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, 2048, shuffle = False)

    model = Modality_CLIP(
        Encoder=Encoder, 
        layers_dims = (
          config.LAYERS_DIM_FIRST, 
          config.LAYERS_DIM_GEX
        ),
        dropout_rates = (
          config.DROPOUT_RATES_FIRST, 
          config.DROPOUT_RATES_GEX
        ),
        dim_mod1 = 134 if mod1 == 'ADT' else config.N_LSI_COMPONENTS_FIRST,
        dim_mod2 = config.N_LSI_COMPONENTS_GEX, 
        output_dim = config.EMBEDDING_DIM,
        T = config.LOG_T,
        swap_rate_1 = 0.,
        swap_rate_2 = 0.)

    optimizer = torch.optim.Adam(model.parameters(), config.LR, weight_decay=config.weight_decay)
    loaders = {
        "train": dataloader_train,
        "valid": dataloader_test,
    }
    runner = scRNARunner()
    
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=config.N_EPOCHS,
        callbacks=[
            dl.OptimizerCallback(metric_key='loss'),
            dl.CheckpointCallback(
                logdir = par['output_pretrain'],
                loader_key='valid',
                metric_key='avg_acc',
                minimize=False,
                use_runner_logdir=False,
                save_n_best=1
            ),
            dl.EarlyStoppingCallback(
                patience=150,
                loader_key='valid',
                metric_key='avg_acc',
                minimize=False,
                min_delta=1e-5),
            dl.LoaderMetricCallback(
                metric=CustomMetric(),
                input_key=['embeddings_first', 'embeddings_second', 'temperature'],
                target_key=['embeddings_second']
            ),
        ],
        verbose=True
    )
    with open(par['output_pretrain'] + '/lsi_GEX_transformer.pickle', 'wb') as f:
        pickle.dump(lsi_transformer_gex, f)
    with open(par['output_pretrain'] + '/lsi_ATAC_transformer.pickle', 'wb') as f:
        pickle.dump(lsi_transformer_atac, f)