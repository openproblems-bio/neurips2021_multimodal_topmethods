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
    'input_explore_mod1': 'output/datasets_explore/cite/cite_gex_processed_training.h5ad',
    'input_explore_mod2': 'output/datasets_explore/cite/cite_gex_processed_training.h5ad',
    'output_pretrain': 'path/to/model'
}
meta = { 'functionality_name': 'cajal_run', 'resources_dir': 'src/predict_modality/methods/cajal/train' }
## VIASH END

input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])

mod_1 = input_train_mod1.var["feature_types"][0]
mod_2 = input_train_mod2.var["feature_types"][0]

os.makedirs(par['output_pretrain'], exist_ok=True)

def atac_de_analysis(path):
    '''get top DA peaks per cell type'''
    adata = sc.read_h5ad(path)
    adata.X = adata.layers['counts']
    adata.X = binarize(adata.X)
    sc.tl.rank_genes_groups(adata, 'cell_type', method='t-test')
    cell_types = adata.obs.cell_type.value_counts().index
    column_names = ['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj', 'cell_type']
    df = pd.DataFrame(columns = column_names)
    for cell_type in cell_types:
        dedf = sc.get.rank_genes_groups_df(adata, group=cell_type)
        dedf['cell_type'] = cell_type
        dedf = dedf.sort_values('scores', ascending=False).iloc[:100]
        df = df.append(dedf, ignore_index=True)
    return df
        
        
def gex_de_analysis(path):
    '''get top DE genes per cell type (multiome)'''
    adata_GEX = sc.read_h5ad(path)
    adata_GEX.X = adata_GEX.layers['counts']
    sc.pp.normalize_per_cell(adata_GEX, counts_per_cell_after=1e6)
    sc.pp.log1p(adata_GEX)
    sc.pp.filter_cells(adata_GEX, min_genes=200)
    sc.pp.filter_genes(adata_GEX, min_cells=3)
    adata_GEX.var['mt'] = adata_GEX.var_names.str.startswith('MT-') 
    sc.pp.calculate_qc_metrics(adata_GEX, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata_GEX = adata_GEX[adata_GEX.obs.n_genes_by_counts < 4000, :]
    sc.pp.normalize_total(adata_GEX, target_sum=1e4)
    sc.pp.log1p(adata_GEX)
    sc.pp.highly_variable_genes(adata_GEX, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pp.scale(adata_GEX, max_value=10)
    sc.tl.rank_genes_groups(adata_GEX, 'cell_type', method='wilcoxon')
    cell_types = adata_GEX.obs.cell_type.value_counts().index
    column_names = ['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj', 'cell_type']
    df = pd.DataFrame(columns = column_names)
    for cell_type in cell_types:
        dedf = sc.get.rank_genes_groups_df(adata_GEX, group=cell_type)
        dedf['cell_type'] = cell_type
        dedf = dedf.sort_values('scores', ascending=False).iloc[:100]
        df = df.append(dedf, ignore_index=True)
    return df

class Hyperparameters:
    #class to store hyperparameters
    def __init__(self, dropout, layer_shapes):
        self.dropout = dropout
        self.layer_shapes = layer_shapes
        self.n_layers = len(layer_shapes)

def build_model(hp, input_shape, output_shape, min_val, max_val):
    '''build models given hyperparameters, input_shape, output_shape, min_val and max_val'''
    model = keras.Sequential()
    model.add(keras.Input(shape = input_shape))
    model.add(keras.layers.Dropout(hp.dropout))
    for i in range(hp.n_layers):
        model.add(keras.layers.Dense(hp.layer_shapes[i],"relu"))
    model.add(keras.layers.Dense(output_shape,None))
    model.add(tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x,min_val,max_val)))
    model.compile(optimizer='adam',
                  loss=keras.losses.MeanSquaredError())
    return model

# define settings
if mod_1 == "GEX" and mod_2 == "ATAC":
    genes = gex_de_analysis(par['input_explore_mod1'])
    genes.to_csv(par['output_pretrain'] + '/DEGs.csv')
    selected_genes = set(genes.names)
    hp = Hyperparameters(0.8, [500, 20, 110, 430, 310])
    epochs = 39

elif mod_1 == "GEX" and mod_2 == "ADT":
    genes1 = gex_de_analysis(par['input_explore_mod1'])
    genes1.to_csv(par['output_pretrain'] + '/DEGs.csv')
    genes2 = pd.read_csv(meta["resources_dir"] + "/ADT_list_df_updated.csv")
    selected_genes = set(genes1.names.append(genes2.gene_name))
    hp = Hyperparameters(0.4, [170, 80, 340])
    epochs = 35

elif mod_1 == "ATAC" and mod_2 == "GEX":
    genes = atac_de_analysis(par['input_explore_mod1'])
    genes.to_csv(par['output_pretrain'] + '/DEGs.csv')
    selected_genes = set(genes.names)
    hp = Hyperparameters(0.8, [320, 360, 620, 440])
    epochs = 38

elif mod_1 == "ADT" and mod_2 == "GEX":
    selected_genes = set(input_train_mod1.var_names)
    hp = Hyperparameters(0.2, [180, 140, 520])
    epochs = 78

#### build and train models given hyperparameters, 
#### training data, a set of genes and number of epochs

# get min and max values for modality 1 and 2
min_val = np.min(input_train_mod2.X)
max_val = np.max(input_train_mod2.X)

#calculate ncounts per cell
train_total = np.sum(input_train_mod1.layers['counts'].toarray(), axis=1)

#implement simple log normalisation if modality 1 is GEX
if mod_1 == "GEX":
    input_train_mod1.X = input_train_mod1.layers['counts']
    sc.pp.normalize_per_cell(input_train_mod1, counts_per_cell_after=1e6)
    sc.pp.log1p(input_train_mod1)

#filter features in training data
subset = selected_genes.intersection(input_train_mod1.var_names)
input_train_mod1 = input_train_mod1[:, list(subset)]

#save features used
with open(par['output_pretrain'] + "/genes.pkl", "wb") as out:
    pickle.dump(input_train_mod1.var_names, out, -1)

#convert to dense
X_train = input_train_mod1.X.toarray()
y_train = input_train_mod2.X.toarray()

#determine batches
train_batches = set(input_train_mod1.obs.batch)

#calculate median and standard deviation total counts per batch
input_train_mod1.obs["batch_median"] = 0
input_train_mod1.obs["batch_sd"] = 0
for batch in train_batches:
    input_train_mod1.obs["batch_median"][input_train_mod1.obs.batch == batch] = np.median(train_total[input_train_mod1.obs.batch == batch])
    input_train_mod1.obs["batch_sd"][input_train_mod1.obs.batch == batch] = np.std(train_total[input_train_mod1.obs.batch == batch])

#add n_counts to training data (multiple times to increase feature's importance) 
for i in range(50):
    X_train = np.column_stack((X_train,train_total))

#add median counts per batch to training data (multiple times to increase feature's importance) 
for i in range(50):
    X_train = np.column_stack((X_train,input_train_mod1.obs["batch_median"]))

#add standard deviation per batch to training data (multiple times to increase feature's importance) 
for i in range(50):
    X_train = np.column_stack((X_train,input_train_mod1.obs["batch_sd"]))

#zscore training data and save means and sds so same transformation can be applied to test data 
X_train = X_train.T
means = np.mean(X_train, axis = 1)
sds = np.std(X_train, axis = 1)
means = means.reshape(len(means), 1)
sds = sds.reshape(len(sds), 1)
info = {"means":means,"sds":sds}

with open(par['output_pretrain'] + "/transformation.pkl", "wb") as out:
    pickle.dump(info, out, -1)

X_train = (X_train - means)/sds
X_train = X_train.T
            
#determine input and output shape
input_shape = X_train.shape[1]
output_shape = y_train.shape[1]

#build model
model = build_model(hp, input_shape, output_shape, min_val, max_val)
history = model.fit(X_train, y_train, shuffle = True, epochs=epochs, batch_size=1000)

# save model
model.save(par['output_pretrain'] + "/model.h5")