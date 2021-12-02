import anndata as ad
import pickle
import torch

from torch.utils.data import DataLoader

import sys

import numpy as np

from scipy.sparse import csc_matrix

## VIASH START
dataset_path = "output/datasets/match_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_"
pretrain_path = "output/pretrain/match_modality/clue/openproblems_bmmc_cite_phase2_rna.clue_train.output_pretrain/"

par = {
    'input_train_mod1': f'{dataset_path}train_mod1.h5ad',
    'input_train_mod2': f'{dataset_path}train_mod2.h5ad',
    'input_test_mod1': f'{dataset_path}test_mod1.h5ad',
    'input_pretrain': pretrain_path,
    'output': 'output.h5ad'
}
meta = {
    'resources_dir': '.',
    'functionality_name': '169769'
}
## VIASH END

sys.path.append(meta['resources_dir'])
from helper_functions import ModelRegressionGex2Adt, ModelRegressionGex2Atac, ModelRegressionAtac2Gex, ModelRegressionAdt2Gex, ModalityMatchingDataset


input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])


mod1 = input_train_mod1.var['feature_types'][0]
mod2 = input_train_mod2.var['feature_types'][0]

if mod1 == 'GEX' and mod2 == 'ADT':
    model = ModelRegressionGex2Adt(256,134)   
    weight = torch.load(par['input_pretrain'] + '/model.pt', map_location='cpu')
    with open(par['input_pretrain'] + '/lsi_transformer.pickle', 'rb') as f:
        lsi_transformer_gex = pickle.load(f)
    
        
    model.load_state_dict(weight)    
    input_test_mod1_ = lsi_transformer_gex.transform(input_test_mod1)

elif mod1 == 'GEX' and mod2 == 'ATAC':
    model = ModelRegressionGex2Atac(256,10000)   
    weight = torch.load(par['input_pretrain'] + '/model.pt', map_location='cpu')
    with open(par['input_pretrain'] + '/lsi_transformer.pickle', 'rb') as f:
        lsi_transformer_gex = pickle.load(f)
    
        
    model.load_state_dict(weight)    
    input_test_mod1_ = lsi_transformer_gex.transform(input_test_mod1)
    
elif mod1 == 'ATAC' and mod2 == 'GEX':
    model = ModelRegressionAtac2Gex(256,13431)   
    weight = torch.load(par['input_pretrain'] + '/model.pt', map_location='cpu')
    with open(par['input_pretrain'] + '/lsi_transformer.pickle', 'rb') as f:
        lsi_transformer_gex = pickle.load(f)
        
    model.load_state_dict(weight)    
    input_test_mod1_ = lsi_transformer_gex.transform(input_test_mod1)

elif mod1 == 'ADT' and mod2 == 'GEX':
    model = ModelRegressionAdt2Gex(134,13953)   
    weight = torch.load(par['input_pretrain'] + '/model.pt', map_location='cpu')
        
    model.load_state_dict(weight)    
    #input_test_mod1_ = lsi_transformer_gex.transform(input_test_mod1)
    input_test_mod1_ = input_test_mod1.to_df()
    
dataset_test = ModalityMatchingDataset(input_test_mod1_, None, is_train=False)
dataloader_test = DataLoader(dataset_test, 32, shuffle = False, num_workers = 4)

outputs = []
model.eval()
with torch.no_grad():
    for x in dataloader_test:
        output = model(x.float())
        outputs.append(output.detach().cpu().numpy())

outputs = np.concatenate(outputs)
outputs[outputs<0] = 0
outputs = csc_matrix(outputs)

adata = ad.AnnData(
    X=outputs,
    obs=input_test_mod1.obs,
    var=input_train_mod2.var,
    uns={
        'dataset_id': input_train_mod1.uns['dataset_id'],
        'method_id': meta['functionality_name'],
    },
)
adata.write_h5ad(par['output'], compression = "gzip")