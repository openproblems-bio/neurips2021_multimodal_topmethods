import logging
import anndata as ad
import sys
from scipy.sparse import csc_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
import numpy as np

logging.basicConfig(level=logging.INFO)

## VIASH START
# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.
par = {
    'input_train_mod1': 'output/datasets/predict_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_mod1.h5ad',
    'input_train_mod2': 'output/datasets/predict_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_mod2.h5ad',
    'input_test_mod1': 'output/datasets/predict_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_test_mod1.h5ad',
    'input_pretrain': 'path/to/model',
    'output': 'output.h5ad'
}
meta = {
    'resources_dir': 'src/predict_modality/methods/AXX/resources'
}
## VIASH END
sys.path.append(meta['resources_dir'])
from predict import predict
from utils import get_y_dim

logging.info('Reading `h5ad` files...')
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])

y_dim,task = get_y_dim(par['input_test_mod1'])
ymean = np.asarray(input_train_mod2.X.mean(axis=0))
if task == 'GEX2ATAC':
    y_pred = ymean*np.ones([input_test_mod1.shape[0],y_dim])
else:
    
    y_pred = predict(ymean,test_data_path=par['input_test_mod1'],
                     folds=[0,1,2],cp=meta['resources_dir'],
                     wp=par['input_pretrain'])

y_pred = csc_matrix(y_pred)

adata = ad.AnnData(
    X=y_pred,
    obs=input_test_mod1.obs,
    var=input_train_mod2.var,
    uns={
        'dataset_id': input_train_mod1.uns['dataset_id'],
        'method_id': "axx",
    },
)

logging.info('Storing annotated data...')
adata.write_h5ad(par['output'], compression = "gzip")
