import os
import logging
import anndata as ad
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import binarize

logging.basicConfig(level=logging.INFO)

## VIASH START
par = {
    'input_train_mod1': 'output/datasets/predict_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_mod1.h5ad',
    'input_train_mod2': 'output/datasets/predict_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_mod2.h5ad',
    'output_pretrain': 'path/to/model'
}

meta = {
    'functionality_name': 'AXX', 
    'resources_dir': 'src/predict_modality/methods/AXX/resources'
}
## VIASH END

import sys
sys.path.append(meta['resources_dir'])
from train import train


input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])

mod_1 = input_train_mod1.var["feature_types"][0]
mod_2 = input_train_mod2.var["feature_types"][0]

os.makedirs(par['output_pretrain'], exist_ok=True)

task = f'{mod_1}2{mod_2}'
train(task,cp=meta['resources_dir'],
      wp=par['output_pretrain'],
      tr1=input_train_mod1,
      tr2=input_train_mod2)