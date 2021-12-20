import os
import sys
import json
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pickle as pk
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import scipy
import scanpy as sc
import tensorflow as tf


## VIASH START
dataset_path = 'output/datasets/joint_embedding/openproblems_bmmc_cite_phase2/openproblems_bmmc_cite_phase2.censor_dataset.output_'
par = {
    'input_mod1': f'{dataset_path}mod1.h5ad',
    'input_mod2': f'{dataset_path}mod2.h5ad',
    'input_explore_mod1': 'output/datasets_explore/cite/cite_gex_processed_training.h5ad',
    'input_explore_mod2': 'output/datasets_explore/cite/cite_adt_processed_training.h5ad',
    'output_pretrain': '...',
    'tf_seed': 46,
    'np_seed': 56
}
meta = { 'resources_dir': 'src/joint_embedding/methods/jae/resources', 'method_id': 'submission_171079' }
## VIASH END

sys.path.append(meta['resources_dir'])
from utils import EarlyStoppingAtMinLoss, JointEmbeddingModel

input_mod1 = ad.read_h5ad(par['input_mod1'])
input_mod2 = ad.read_h5ad(par['input_mod2'])
print(input_mod1.shape, input_mod2.shape)

os.makedirs(par['output_pretrain'], exist_ok=True)

##########################################
##             PCA PRETRAIN             ##
##########################################

#scale and log transform
random_seed = 123
scale = 1e4
n_components_mod1, n_components_mod2 = 100, 100

mod1 = input_mod1.var["feature_types"][0]
mod2 = input_mod2.var["feature_types"][0]

if mod1 != "ADT":
    mod1_data = scale * normalize(input_mod1.X,norm='l1', axis=1)
    mod1_data = sp.csr_matrix.log1p(mod1_data) / np.log(10)

    mod1_reducer = TruncatedSVD(n_components=n_components_mod1, random_state=random_seed)
    mod1_reducer.fit(mod1_data)
    pca_data_mod1 = mod1_reducer.transform(mod1_data)
    #print('multiome 1 done',pca_data_mod1.shape)
    pk.dump(mod1_reducer, open(os.path.join(par['output_pretrain'], "svd_mod1.pkl"),"wb"))

    del mod1_data, pca_data_mod1

if mod2 != "ADT":
    mod2_data = scale * normalize(input_mod2.X,norm='l1', axis=1)
    mod2_data = sp.csr_matrix.log1p(mod2_data) / np.log(10)

    mod2_reducer = TruncatedSVD(n_components=n_components_mod2, random_state=random_seed)
    mod2_reducer.fit(mod2_data)
    pca_data_mod2 = mod2_reducer.transform(mod2_data)
    #print('multiome 2 done',pca_data_mod2.shape)
    pk.dump(mod2_reducer, open(os.path.join(par['output_pretrain'], "svd_mod2.pkl"),"wb"))

    del mod2_data, pca_data_mod2

del input_mod1, input_mod2

##########################################
##           TRAIN AUTOENCODER          ##
##########################################

'''Pretrain with only exploration data (with cell type label, cell cycle scores, etc)
NOTE:
    The loss function for each epoch will be printed.
    Please change the par path to the right location of explration data
Output:
    The best pretrained model (multiome.h5 or cite.h5) will be saved to disk under the current path (./).
    The par['output'] recorded the joint embedding of exploration data.
'''

tf_seed = par['tf_seed']
np_seed = par['np_seed']
suffix = str(tf_seed)+'_'+str(np_seed)

np.random.seed(np_seed)
tf.random.set_seed(tf_seed)

ad_mod1 = ad.read_h5ad(par['input_explore_mod1'])
ad_mod2 = ad.read_h5ad(par['input_explore_mod2'])
print(ad_mod1.shape, ad_mod2.shape)
mod1_obs = ad_mod1.obs
mod1_uns = ad_mod1.uns

ad_mod2_var = ad_mod2.var

mod1_mat = ad_mod1.layers["counts"]
mod2_mat = ad_mod2.layers["counts"]

#exploration data (with labels) gene expression is not log1p normalized
mod1_mat = scipy.sparse.csr_matrix.log1p(mod1_mat)
print(np.max(mod1_mat))

cell_cycle_genes = [
    'MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2',\
    'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', \
    'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP',\
    'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1', \
    'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', \
    'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2', 'CDC45', \
    'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', \
    'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8', \
    'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', \
    'TOP2A', 'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', \
    'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', \
    'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', \
    'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', \
    'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', \
    'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', \
    'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', \
    'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA']

print(mod1_mat.shape, mod2_mat.shape)

def preprocess(mod1_data, mod2_data, scale=1e4):
    # n_components_mod1, n_components_mod2, mod1_reducer and mod2_reducer 
    # are still in memory from before

    mod1_data = scale * normalize(mod1_data,norm='l1', axis=1)
    mod2_data = scale * normalize(mod2_data,norm='l1', axis=1)
    mod1_data = scipy.sparse.csr_matrix.log1p(mod1_data) / np.log(10)
    mod2_data = scipy.sparse.csr_matrix.log1p(mod2_data) / np.log(10)
    
    pca_data_mod1 = mod1_reducer.transform(mod1_data)

    if ad_mod2_var['feature_types'][0] == 'ADT':
        pca_data_mod2 = mod2_data.toarray()
    else:
        #mod2_reducer.fit(mod2_data)
        pca_data_mod2 = mod2_reducer.transform(mod2_data)
    return pca_data_mod1, pca_data_mod2

mod1_pca, mod2_pca = preprocess(mod1_mat, mod2_mat)

del mod1_mat, mod2_mat

print('load data and pca done', mod1_pca.shape, mod2_pca.shape)

pca_combined = np.concatenate([mod1_pca, mod2_pca],axis=1)

print(pca_combined.shape)

del mod1_pca, mod2_pca

cell_type_labels = mod1_obs['cell_type']
batch_ids = mod1_obs['batch']
phase_labels = mod1_obs['phase']
nb_cell_types = len(np.unique(cell_type_labels))
nb_batches = len(np.unique(batch_ids))
nb_phases = len(np.unique(phase_labels))-1 # 2
c_labels = np.array([list(np.unique(cell_type_labels)).index(item) for item in cell_type_labels])
b_labels = np.array([list(np.unique(batch_ids)).index(item) for item in batch_ids])
p_labels = np.array([list(np.unique(phase_labels)).index(item) for item in phase_labels])
#0:G1, 1:G2M, 2: S, only consider the last two
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]
sc.pp.log1p(ad_mod1)
sc.pp.scale(ad_mod1)
sc.tl.score_genes_cell_cycle(ad_mod1, s_genes=s_genes, g2m_genes=g2m_genes)
S_scores = ad_mod1.obs['S_score'].values
G2M_scores = ad_mod1.obs['G2M_score'].values
phase_scores = np.stack([S_scores,G2M_scores]).T #(nb_cells, 2)


pca_train, pca_test, c_train_labels, c_test_labels, b_train_labels, b_test_labels, p_train_labels, p_test_labels, phase_train_scores, phase_test_scores = train_test_split(pca_combined, c_labels, b_labels, p_labels,phase_scores, test_size=0.1, random_state=42)
print(pca_train.shape, c_train_labels.shape, b_train_labels.shape, p_train_labels.shape, phase_train_scores.shape)
print(pca_test.shape, c_test_labels.shape, b_test_labels.shape, p_test_labels.shape, phase_test_scores.shape)
X_train = pca_train
#Y_train = [pca_train, c_train_labels, b_train_labels, p_train_labels]
Y_train = [pca_train, c_train_labels, b_train_labels, phase_train_scores]

X_test = pca_test
#Y_test = [pca_test, c_test_labels, b_test_labels, p_test_labels]
Y_test = [pca_test, c_test_labels, b_test_labels, phase_test_scores]

print(nb_cell_types, nb_batches, nb_phases)
print(ad_mod2_var['feature_types'][0])

hidden_units = [150, 120, 100, nb_cell_types+nb_batches+nb_phases+5]

params = {
    'dim' : pca_combined.shape[1],
    'lr': 1e-4,
    'hidden_units' : hidden_units,
    'nb_layers': len(hidden_units),
    'nb_cell_types': nb_cell_types,
    'nb_batches': nb_batches,
    'nb_phases': nb_phases,
    'use_batch': True
}

with open(os.path.join(par['output_pretrain'], 'hyperparams.json'), 'w') as file:
     file.write(json.dumps(params))

print('Model hyper parameters:', params)

def random_classification_loss(y_true, y_pred):
    return tf.keras.metrics.categorical_crossentropy(tf.ones_like(y_pred)/nb_batches, y_pred, from_logits=True)

model = JointEmbeddingModel(params)

model.compile(tf.keras.optimizers.Adam(learning_rate = params["lr"]), 
            loss = [tf.keras.losses.MeanSquaredError(), 
                    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    random_classification_loss,
                    tf.keras.losses.MeanSquaredError()
                    ],
            loss_weights=[0.7, 0.2, 0.05, 0.05], 
            run_eagerly=True)

callbacks = [EarlyStoppingAtMinLoss(patience=5),
            tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(par['output_pretrain'],"weights.h5"),
                             monitor='val_loss', save_weights_only=True)]

model.fit(x=X_train, y=Y_train,
                epochs = 500,
                batch_size = 32,
                shuffle=True,
                callbacks = callbacks,
                validation_data=(X_test, Y_test),
                max_queue_size = 100, workers = 28, use_multiprocessing = True)

print('Start evaluation')
eval_results = model.evaluate(X_test, Y_test, batch_size=128)
print('Total loss, loss1, loss2, loss3, loss4:',eval_results)

f_out = open(os.path.join(par['output_pretrain'],'train.log'),'a+')
f_out.write('%s\t%.4f\t%.4f\t%.4f\t%.4f\n'%(suffix, eval_results[1], eval_results[2], eval_results[3], eval_results[4]))
f_out.close()
