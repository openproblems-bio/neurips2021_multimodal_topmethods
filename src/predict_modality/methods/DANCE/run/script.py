import logging
import anndata as ad
import numpy as np
import sys
from scipy.sparse import csc_matrix


logging.basicConfig(level=logging.INFO)

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
    'functionality_name': '171129'
}
## VIASH END

logging.info('Reading `h5ad` files...')
train_mod1 = ad.read_h5ad(par['input_train_mod1'])
mod1 = train_mod1.var['feature_types'][0]
dataset_id = train_mod1.uns['dataset_id']
input_train_mod1 = train_mod1.X

train_mod2 = ad.read_h5ad(par['input_train_mod2'])
var = train_mod2.var
mod2 = train_mod2.var['feature_types'][0]
input_train_mod2 = train_mod2.X

test_mod1 = ad.read_h5ad(par['input_test_mod1'])
obs = test_mod1.obs
input_test_mod1 = test_mod1.X

if mod1 == 'GEX':
    sys.path.append(meta['resources_dir'])
    from graph_util import graph_construction
    
    import torch

    # # This will get passed to the method
    FEATURE_SIZE = train_mod1.shape[1]
    OUTPUT_SIZE = train_mod2.shape[1]
    TRAIN_SIZE = train_mod1.shape[0]
    TEST_SIZE = test_mod1.shape[0]
    
    g, bf = graph_construction(meta, train_mod1, train_mod2, test_mod1, pretrain_path=par['input_pretrain'])
    
    def evaluate(mod):
        mod.eval()
        with torch.no_grad():
            logits = mod(g, bf)
            logits = logits[-TEST_SIZE:]
            return logits
    
    if mod2 == 'ADT':
        y_pred = []
        model = torch.load(par['input_pretrain'] + '/f_alpha_conv4_mean_fullbatch_12000_phase2_inductive_batch_speration.pkl')
        y_pred.append(evaluate(model).numpy())
        
        model = torch.load(par['input_pretrain'] + '/bf_alpha_conv4_mean_fullbatch_10000_phase2_inductive_gex2adt_2.pkl')
        y_pred.append(evaluate(model).numpy())
        
        model = torch.load(par['input_pretrain'] + '/bf_alpha_conv4_mean_fullbatch_12000_phase2_inductive_gex2adt_sep_2.pkl')
        y_pred.append(evaluate(model).numpy())
        
        model = torch.load(par['input_pretrain'] + '/bf_alpha_conv4_mean_fullbatch_15000_phase2_inductive.pkl')
        y_pred.append(evaluate(model).numpy())
        
        y_pred = csc_matrix((y_pred[0]+y_pred[1]+y_pred[2]+y_pred[3])/4)
    
    elif mod2 == 'ATAC':
        y_pred = []
        model = torch.load(par['input_pretrain'] + '/bf_alpha_conv4_mean_fullbatch_8000_phase2_inductive_gex2atac_3.pkl')
        y_pred.append(evaluate(model).numpy())
        
        model = torch.load(par['input_pretrain'] + '/bf_alpha_conv4_mean_fullbatch_8000_phase2_inductive_gex2atac_2.pkl')
        y_pred.append(evaluate(model).numpy())
        
        model = torch.load(par['input_pretrain'] + '/bf_alpha_conv4_mean_fullbatch_8000_phase2_inductive_gex2atac.pkl')
        y_pred.append(evaluate(model).numpy())
        
        model = torch.load(par['input_pretrain'] + '/bf_alpha_conv4_mean_fullbatch_10000_phase2_inductive_gex2atac.pkl')
        y_pred.append(evaluate(model).numpy())
        
        y_pred = csc_matrix((y_pred[0]+y_pred[1]+y_pred[2]+y_pred[3])/4)

elif mod1=='ATAC' and mod2=='GEX':
    y_pred = csc_matrix(np.tile(np.mean(input_train_mod2.toarray(), 0), (input_test_mod1.shape[0], 1)))
        
else:
    sys.path.append(meta['resources_dir'])
    from baseline import baseline_linear

    input_train_mod1 = train_mod1[train_mod1.obs['batch']!='s3d1'].X
    input_train_mod2 = train_mod2[train_mod2.obs['batch']!='s3d1'].X
    y_pred = csc_matrix(baseline_linear(input_train_mod1, input_train_mod2, input_test_mod1))

adata = ad.AnnData(
    X=y_pred,
    obs=obs,
    var=var,
    uns={
        'dataset_id': dataset_id,
        'method_id': meta['functionality_name'],
    },
)

logging.info('Storing annotated data...')
adata.write_h5ad(par['output'], compression = "gzip")