import logging
import anndata as ad
import pickle
import numpy as np

from scipy.sparse import csc_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

import tensorflow as tf
import scanpy as sc

logging.basicConfig(level=logging.INFO)

## VIASH START
par = {
    'input_train_mod1': 'sample_data/predict_modality/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod1.h5ad',
    'input_train_mod2': 'sample_data/predict_modality/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod2.h5ad',
    'input_test_mod1': 'sample_data/predict_modality/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.test_mod1.h5ad',
    'input_pretrain': 'path/to/model',
    'output': 'output.h5ad'
}
meta = { 'functionality_name': 'cajal_run' }
## VIASH END

logging.info('Reading `h5ad` files...')
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])

#get modalities
mod_1 = input_train_mod1.var["feature_types"][0]
mod_2 = input_train_mod2.var["feature_types"][0]

try:

    test_total = np.sum(input_test_mod1.layers['counts'].toarray(), axis=1)

    if mod_1 == "GEX":
        input_test_mod1.X = input_test_mod1.layers['counts']
        sc.pp.normalize_per_cell(input_test_mod1, counts_per_cell_after=1e6)
        sc.pp.log1p(input_test_mod1)
    
    with open(par["input_pretrain"] + "/genes.pkl", "rb") as f:
            genes = pickle.load(f)
            input_test_mod1 = input_test_mod1[:,genes]
    
    if mod_1 == "GEX":
        input_train_mod1.X = input_train_mod1.layers['counts']
        sc.pp.normalize_per_cell(input_train_mod1, counts_per_cell_after=1e6)
        sc.pp.log1p(input_train_mod1)

    X_test = input_test_mod1.X.toarray()

    test_batches = set(input_test_mod1.obs.batch)

    input_test_mod1.obs["batch_median"] = 0

    input_test_mod1.obs["batch_sd"] = 0

    for batch in test_batches:
        input_test_mod1.obs["batch_median"][input_test_mod1.obs.batch == batch] = np.median(test_total[input_test_mod1.obs.batch == batch])
        input_test_mod1.obs["batch_sd"][input_test_mod1.obs.batch == batch] = np.std(test_total[input_test_mod1.obs.batch == batch])


    for i in range(50):
        X_test = np.column_stack((X_test,test_total))

    for i in range(50):
        X_test = np.column_stack((X_test,input_test_mod1.obs["batch_median"]))

    for i in range(50):
        X_test = np.column_stack((X_test,input_test_mod1.obs["batch_sd"]))

    with open(par["input_pretrain"] + "/transformation.pkl", "rb") as f:
            info = pickle.load(f)

    X_test = X_test.T
    X_test = (X_test - info["means"])/info["sds"]
    X_test = X_test.T


    #load pretrained model for correct modalities
    model = tf.keras.models.load_model(par["input_pretrain"] + "/model.h5")

    #make predictions for y
    y_pred = model.predict(X_test)

except:
    logging.info("Error! Falling back to default")
    input_train = ad.concat(
        { 'train': input_train_mod1, 'test': input_test_mod1},
        axis=0,
        join="outer",
        label="group",
        fill_value=0,
        index_unique="-"
    )


    # Do PCA on the input data
    logging.info('Performing dimensionality reduction on modality 1 values...')
    embedder_mod1 = TruncatedSVD(n_components=50)
    mod1_pca = embedder_mod1.fit_transform(input_train.X)

    logging.info('Performing dimensionality reduction on modality 2 values...')
    embedder_mod2 = TruncatedSVD(n_components=50)
    mod2_pca = embedder_mod2.fit_transform(input_train_mod2.X)

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
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # Project the predictions back to the modality 2 feature space
    y_pred = y_pred @ embedder_mod2.components_

#convert to sparse matrix
y_pred = csc_matrix(y_pred)

adata = ad.AnnData(
    X=y_pred,
    obs=input_test_mod1.obs,
    var=input_train_mod2.var,
    uns={
        'dataset_id': input_train_mod1.uns['dataset_id'],
        'method_id': "cajal",
    },
)


logging.info('Storing annotated data...')
adata.write_h5ad(par['output'], compression = "gzip")
