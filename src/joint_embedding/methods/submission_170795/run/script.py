import logging
import anndata as ad
import numpy as np

from sklearn.decomposition import TruncatedSVD

logging.basicConfig(level=logging.INFO)

## VIASH START
dataset_path = "output/datasets/joint_embedding/openproblems_bmmc_cite_phase2/openproblems_bmmc_cite_phase2.censor_dataset.output_"

par = {
    'input_mod1': f'{dataset_path}mod1.h5ad',
    'input_mod2': f'{dataset_path}mod2.h5ad',
    'output': 'output.h5ad'
}
meta = {
    'resources_dir': '.',
    'functionality_name': 'submission_170795'
}
## VIASH END

def normalize(arr):
    arr_sd = np.std(arr, axis=1).reshape(-1, 1)
    arr_mean = np.mean(arr, axis=1).reshape(-1, 1)
    return (arr - arr_mean) / arr_sd

logging.info('Reading `h5ad` files...')
ad_mod1 = ad.read_h5ad(par['input_mod1'])
ad_mod2 = ad.read_h5ad(par['input_mod2'])

logging.info('Determine parameters by the modalities')
mod1_type = ad_mod1.var.feature_types[0].upper()
mod2_type = ad_mod2.var.feature_types[0].upper()

if mod1_type == "GEX" and mod2_type == "ADT":
    n_mod1 = 73
    n_mod2 = 27

elif mod1_type == "ADT" and mod2_type == "GEX":
    n_mod1 = 27
    n_mod2 = 73

elif mod1_type == "GEX" and mod2_type == "ATAC":
    n_mod1 = 38
    n_mod2 = 62

elif mod1_type == "ATAC" and mod2_type == "GEX":
    n_mod1 = 62
    n_mod2 = 38

else:
    n_mod1 = 50
    n_mod2 = 50

logging.info('Performing dimensionality reduction on modality 1 values...')
embedder_mod1 = TruncatedSVD(n_components=n_mod1)
mod1_pca = embedder_mod1.fit_transform(ad_mod1.X)
mod1_obs = ad_mod1.obs
mod1_uns = ad_mod1.uns
del ad_mod1

logging.info('Performing dimensionality reduction on modality 2 values...')
embedder_mod1 = TruncatedSVD(n_components=n_mod2)
mod2_pca = embedder_mod1.fit_transform(ad_mod2.X)
del ad_mod2

logging.info('Concatenating datasets')
pca_combined = np.concatenate([normalize(mod1_pca), normalize(mod2_pca)], axis=1)

logging.info('Storing output to file')
adata = ad.AnnData(
    X=pca_combined,
    obs=mod1_obs,
    uns={
        'dataset_id': mod1_uns['dataset_id'],
        'method_id': meta['functionality_name'],
    },
)
adata.write_h5ad(par['output'], compression="gzip")
