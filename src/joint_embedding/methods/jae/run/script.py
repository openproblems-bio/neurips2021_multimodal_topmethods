import os
import sys
import logging
import json
import anndata as ad
import numpy as np
from sklearn.preprocessing import normalize
import tensorflow as tf
import pickle as pk
import scipy

logging.basicConfig(level=logging.INFO)

## VIASH START
dataset_path = 'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.'

par = {
    'input_mod1': dataset_path + 'mod1.h5ad',
    'input_mod2': dataset_path + 'mod2.h5ad',
    'input_pretrain': '...',
    'output': 'output.h5ad',
}

meta = { 'resources_dir': '.', 'functionality_name': 'submission_171079' }
## VIASH END

sys.path.append(meta['resources_dir'])
from utils import JointEmbeddingModel

logging.info('Reading `h5ad` files...')
ad_mod1 = ad.read_h5ad(par['input_mod1'])
ad_mod2 = ad.read_h5ad(par['input_mod2'])
mod1_obs = ad_mod1.obs
mod1_uns = ad_mod1.uns

ad_mod2_var = ad_mod2.var

mod_type = ad_mod2_var['feature_types'][0]

mod1_mat = ad_mod1.layers["counts"]
mod2_mat = ad_mod2.layers["counts"]

del ad_mod2, ad_mod1

if mod_type == 'ATAC':
    mod1_svd = pk.load(open(os.path.join(par['input_pretrain'], 'svd_mod1.pkl'),'rb'))
    mod2_svd = pk.load(open(os.path.join(par['input_pretrain'], 'svd_mod2.pkl'),'rb'))
else:
    mod1_svd = pk.load(open(os.path.join(par['input_pretrain'], 'svd_mod1.pkl'),'rb'))
    mod2_svd = None

def svd_transform(mod1_data, mod2_data, mod1_svd, mod2_svd, scale=1e4):
    mod1_data = scale * normalize(mod1_data, norm='l1', axis=1)
    mod2_data = scale * normalize(mod2_data, norm='l1', axis=1)
    mod1_data = scipy.sparse.csr_matrix.log1p(mod1_data) / np.log(10)
    mod2_data = scipy.sparse.csr_matrix.log1p(mod2_data) / np.log(10)
    pca_data_mod1 = mod1_svd.transform(mod1_data)

    if mod_type == 'ADT':
        pca_data_mod2 = mod2_data.toarray()
    else:
        pca_data_mod2 = mod2_svd.transform(mod2_data)
    return pca_data_mod1, pca_data_mod2

mod1_pca, mod2_pca = svd_transform(mod1_mat, mod2_mat, mod1_svd, mod2_svd)

del mod1_mat, mod2_mat

pca_combined = np.concatenate([mod1_pca, mod2_pca],axis=1)
del mod1_pca, mod2_pca

if mod_type == 'ATAC':
    epochs = 2
else:
    epochs = 1

coeff = [1.0, 0.0, 0.0, 0.0]

with open(os.path.join(par['input_pretrain'], 'hyperparams.json'), 'r') as file:
     params = json.load(file)

mymodel = JointEmbeddingModel(params)
mymodel(np.zeros((2, params['dim'])))

mymodel.compile(tf.keras.optimizers.Adam(learning_rate = params["lr"]), 
            loss = [tf.keras.losses.MeanSquaredError(), 
                    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    tf.keras.losses.MeanSquaredError()
                    ],
            loss_weights=coeff, run_eagerly=True)

#load pretrain model
mymodel.load_weights(os.path.join(par['input_pretrain'], 'weights.h5'))


X_train = pca_combined
c_fakes = np.random.randint(low=0, high=params['nb_cell_types'],size=pca_combined.shape[0])
b_fakes = np.random.randint(low=0, high=params['nb_batches'],size=pca_combined.shape[0])
p_fakes = np.random.randint(low=0, high=params['nb_phases'],size=pca_combined.shape[0])
Y_train = [pca_combined, c_fakes, b_fakes, p_fakes]

#finetune on the test data
mymodel.fit(x=X_train, y=Y_train,
            epochs = epochs,
            batch_size = 32,
            shuffle=True)

embeds = mymodel.encoder.predict(pca_combined)
print(embeds.shape)

adata = ad.AnnData(
    X=embeds,
    obs=mod1_obs,
	uns={
        'dataset_id': mod1_uns['dataset_id'],
        'method_id': meta['functionality_name'],
    },
)
adata.write_h5ad(par['output'], compression="gzip")
