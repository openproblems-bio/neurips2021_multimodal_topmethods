# Dependencies:
# pip: scikit-learn, anndata, scanpy
#
# Python starter kit for the NeurIPS 2021 Single-Cell Competition.
# Parts with `TODO` are supposed to be changed by you.
#
# More documentation:
#
# https://viash.io/docs/creating_components/python/

import logging
import anndata as ad

import numpy as np
from scipy.sparse import csc_matrix

from cuml.decomposition import TruncatedSVD
from cuml.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)

## VIASH START
# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.
par = {
    'input_train_mod1': 'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod1.h5ad',
    'input_train_mod2': 'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod2.h5ad',
    'input_test_mod1': 'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.test_mod1.h5ad',
    'output': 'output.h5ad',
}
## VIASH END

# TODO: change this to the name of your method
method_id = "python_starter_kit"

logging.info('Reading `h5ad` files...')
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])

input_train = ad.concat(
    {"train": input_train_mod1, "test": input_test_mod1},
    axis=0,
    join="outer",
    label="group",
    fill_value=0,
    index_unique="-"
)

# TODO: implement own method

# Do PCA on the input data
logging.info('Performing dimensionality reduction on modality 1 values...')
embedder_mod1 = TruncatedSVD(n_components=50)
mod1_pca = embedder_mod1.fit_transform(input_train.X.toarray())

logging.info('Performing dimensionality reduction on modality 2 values...')
embedder_mod2 = TruncatedSVD(n_components=50)
mod2_pca = embedder_mod2.fit_transform(input_train_mod2.X.toarray())

# split dimred back up
X_train = mod1_pca[input_train.obs['group'] == 'train']
X_test = mod1_pca[input_train.obs['group'] == 'test']
y_train = mod2_pca

assert len(X_train) + len(X_test) == len(mod1_pca)

# Get all responses of the training data set to fit the
# KNN regressor later on.
#
# Make sure to use `toarray()` because the output might
# be sparse and `KNeighborsRegressor` cannot handle it.

logging.info('Running Linear regression...')

reg = LinearRegression()

# Train the model on the PCA reduced modality 1 and 2 data
y_pred = []
for i in range(y_train.shape[1]):
    reg.fit(X_train, y_train[:,i])
    y_pred.append(reg.predict(X_test))
y_pred = np.array(y_pred).T

# Project the predictions back to the modality 2 feature space
y_pred = y_pred @ embedder_mod2.components_.to_output('numpy')

# Store as sparse matrix to be efficient. Note that this might require
# different classifiers/embedders before-hand. Not every class is able
# to support such data structures.
y_pred = csc_matrix(y_pred)

adata = ad.AnnData(
    X=y_pred,
    obs=input_test_mod1.obs,
    var=input_train_mod2.var,
    uns={
        'dataset_id': input_train_mod1.uns['dataset_id'],
        'method_id': method_id,
    },
)

logging.info('Storing annotated data...')
adata.write_h5ad(par['output'], compression = "gzip")

