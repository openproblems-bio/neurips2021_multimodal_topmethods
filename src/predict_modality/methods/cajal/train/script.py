import logging
import anndata as ad
#import pickle5 as pickle
import pickle
import numpy as np

from scipy.sparse import csc_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import math
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

## VIASH START
par = {
    'input_train_mod1': 'sample_data/predict_modality/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod1.h5ad',
    'input_train_mod2': 'sample_data/predict_modality/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod2.h5ad',
    'output_pretrain': 'path/to/model'
}
meta = { 'functionality_name': 'cajal_run' }
## VIASH END


def highly_variable(adata):
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_hvar = adata[:,adata.var.highly_variable]
    return adata_hvar

def highly_variable_500_plus(adata):
    sc.pp.highly_variable_genes(adata,n_top_genes = 500, flavor = "cell_ranger")
    genes = pd.read_csv("ADT_list_df_updated.csv")
    adata.var.highly_variable[adata.var_names.isin(genes.gene_name)] = True
    adata_hvar = adata[:,adata.var.highly_variable]
    return adata_hvar

def select_genes(adata):
    genes = pd.read_csv("ADT_list_df_updated.csv")
    selected_genes= genes.gene_name[genes.gene_name.isin(adata.var_names)]
    adata_sel = adata[:,selected_genes]
    return adata_sel

def highly_variable_1000_plus(adata):
    sc.pp.highly_variable_genes(adata,n_top_genes = 1000, flavor = "cell_ranger")
    genes = pd.read_csv("ADT_list_df_updated.csv")
    adata.var.highly_variable[adata.var_names.isin(genes.gene_name)] = True
    adata_hvar = adata[:,adata.var.highly_variable]
    return adata_hvar


def highly_variable_2000_plus(adata):
    sc.pp.highly_variable_genes(adata,n_top_genes = 2000, flavor = "cell_ranger")
    genes = pd.read_csv("ADT_list_df_updated.csv")
    adata.var.highly_variable[adata.var_names.isin(genes.gene_name)] = True
    adata_hvar = adata[:,adata.var.highly_variable]
    return adata_hvar

def highly_variable_2000(adata):
    sc.pp.highly_variable_genes(adata,n_top_genes = 2000, flavor = "cell_ranger")
    adata_hvar = adata[:,adata.var.highly_variable]
    return adata_hvar

def no_filtering(adata):
    return adata

def de_genes_gex_adt(adata):
    genes1 = pd.read_csv("GEX_ADT_DEGs_wilcoxon.csv")
    genes2 = pd.read_csv("ADT_list_df_updated.csv")
    genes = genes1.names.append(genes2.gene_name)
    selected_genes= genes[genes.isin(adata.var_names)]
    selected_genes = list(set(selected_genes))
    adata_sel = adata[:,selected_genes]
    return adata_sel


def de_genes_gex_atac(adata):
    genes = pd.read_csv("GEX_ATAC_DEGs_wilcoxon.csv")
    selected_genes= genes.names[genes.names.isin(adata.var_names)]
    selected_genes = list(set(selected_genes))
    adata_sel = adata[:,selected_genes]
    return adata_sel

def da_peaks_atac_gex(adata):
    genes = pd.read_csv("ATAC_DEGs_t_test.csv")
    selected_genes= genes.names[genes.names.isin(adata.var_names)]
    selected_genes = list(set(selected_genes))
    adata_sel = adata[:,selected_genes]
    return adata_sel
    


def build_model(hp, input_shape, output_shape, min_val, max_val):
    model = keras.Sequential()
    model.add(keras.Input(shape = input_shape))
    model.add(keras.layers.Dropout(hp.get("dropout")))
    for i in range(hp.get("n_layers")):
        model.add(keras.layers.Dense(hp.get(f"layer_{i}_units"),"relu"))
    model.add(keras.layers.Dense(output_shape,None))
    model.add(tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x,min_val,max_val)))
    model.compile(optimizer='adam',
                  loss=keras.losses.MeanSquaredError())
    return model

def build_model_adt(input_shape, output_shape, min_val, max_val):
    model = keras.Sequential()
    model.add(keras.Input(shape = input_shape))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(180,"relu"))
    model.add(keras.layers.Dense(140,"relu"))
    model.add(keras.layers.Dense(520,"relu"))
    model.add(keras.layers.Dense(output_shape,None))
    model.add(tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x,min_val,max_val)))
    model.compile(optimizer='adam',
                  loss=keras.losses.MeanSquaredError())
    return model

# run 1
tuner = "RandomSearch/GEX_ATAC_DEGs_batch_info_tuner_1637878782.pkl"
model_num = 2
input_train_mod1 = par["input_train_mod1"]
input_train_mod2 = par["input_train_mod2"]
filter_func = de_genes_gex_atac
epochs = 39
name = "batch2"

# run 2
tuner = "RandomSearch/ATAC_GEX_DEGs_batch_info_tuner_1637884812.pkl"
model_num = 1
input_train_mod1 = par["input_train_mod1"]
input_train_mod2 = par["input_train_mod2"]
filter_func = da_peaks_atac_gex
epochs = 38
name = "batch0"

## 

input_train_mod1 = ad.read_h5ad(input_train_mod1)
input_train_mod2 = ad.read_h5ad(input_train_mod2)

mod_1 = input_train_mod1.var["feature_types"][0]
mod_2 = input_train_mod2.var["feature_types"][0]

if mod_1 != "ADT":
    with open(tuner, "rb") as f:
        tuner = pickle.load(f)
        
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials = 5)
    hp = best_hyperparameters[model_num]

min_val = np.min(input_train_mod2.X)
max_val = np.max(input_train_mod2.X)

train_total = np.sum(input_train_mod1.layers['counts'].toarray(), axis=1)

if mod_1 == "GEX":
    input_train_mod1.X = input_train_mod1.layers['counts']

    sc.pp.normalize_per_cell(input_train_mod1, counts_per_cell_after=1e6)
    sc.pp.log1p(input_train_mod1)


input_train_mod1 = filter_func(input_train_mod1)

with open("models_batch_info_3/" + mod_1 + '_' + mod_2 +"_" + name + "_genes.pkl", "wb") as out:
    pickle.dump(input_train_mod1.var_names, out, -1)
X_train = input_train_mod1.X.toarray()
y_train = input_train_mod2.X.toarray()


train_batches = set(input_train_mod1.obs.batch)


input_train_mod1.obs["batch_median"] = 0


input_train_mod1.obs["batch_sd"] = 0


for batch in train_batches:
    input_train_mod1.obs["batch_median"][input_train_mod1.obs.batch == batch] = np.median(train_total[input_train_mod1.obs.batch == batch])
    input_train_mod1.obs["batch_sd"][input_train_mod1.obs.batch == batch] = np.std(train_total[input_train_mod1.obs.batch == batch])

for i in range(50):
    X_train = np.column_stack((X_train,train_total))

for i in range(50):
    X_train = np.column_stack((X_train,input_train_mod1.obs["batch_median"]))

for i in range(50):
    X_train = np.column_stack((X_train,input_train_mod1.obs["batch_sd"]))

X_train = X_train.T

means = np.mean(X_train, axis = 1)
sds = np.std(X_train, axis = 1)
means = means.reshape(len(means), 1)
sds = sds.reshape(len(sds), 1)
info = {"means":means,"sds":sds}
with open("models_batch_info_3/" + mod_1 + '_' + mod_2 + "_" + name + "_transformation.pkl", "wb") as out:
    pickle.dump(info, out, -1)
X_train = (X_train - means)/sds

X_train = X_train.T

input_shape = X_train.shape[1]
output_shape = y_train.shape[1]
if mod_1 == "ADT":
    model = build_model_adt(input_shape, output_shape, min_val, max_val)
else:
    model = build_model(hp, input_shape, output_shape, min_val, max_val)
model.compile(optimizer='adam',
        loss=keras.losses.MeanSquaredError())
history = model.fit(X_train, y_train, shuffle = True, epochs=epochs, batch_size=1000)
model.save("models_batch_info_3/" + mod_1 + '_' + mod_2 + "_" + name + ".h5")





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
        'method_id': meta["functionality_name"],
    },
)


logging.info('Storing annotated data...')
adata.write_h5ad(par['output'], compression = "gzip")
