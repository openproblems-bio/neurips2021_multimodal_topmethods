import logging
import anndata as ad
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')
import scanpy as sc
#from keras import backend as K
from tensorflow.keras.constraints import Constraint
import tensorflow.keras.backend as K

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import warnings
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.compat.v1.random.set_random_seed(2)


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


warnings.filterwarnings('ignore')

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


logging.info('Reading `h5ad` files...')
ad_mod1 = ad.read_h5ad(par['input_mod1'])
ad_mod2 = ad.read_h5ad(par['input_mod2'])

# high variable gene calculation
min_cells = int(ad_mod2.shape[0] * 0.03)
sc.pp.highly_variable_genes(ad_mod1, batch_key ='batch', subset = True)
sc.pp.filter_genes(ad_mod2, min_cells=min_cells)

ad_mod_1 = ad_mod1[:, ad_mod1.var.highly_variable]

## Convert to  csv for AE training
scRNAseq1 = ad_mod_1.X.toarray()
scRNAseq2 = ad_mod2.X.toarray()


class WeightsOrthogonalityConstraint(Constraint):
    def __init__(self, encoding_dim, weightage = 1.0, axis = 0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.axis = axis
        
    def weights_orthogonality(self, w):
        if(self.axis==1):
            w = K.transpose(w)
        if(self.encoding_dim > 1):
            m = K.dot(K.transpose(w), w) - K.eye(self.encoding_dim)
            return self.weightage * K.sqrt(K.sum(K.square(m)))
        else:
            m = K.sum(w ** 2) - 1.
            return m

    def __call__(self, w):
        return self.weights_orthogonality(w)


# Input Layer
ncol_scRNAseq1 = scRNAseq1.shape[1]
input_dim_scRNAseq1 = Input(shape = (ncol_scRNAseq1, ), name = "scRNAseq1")
ncol_scRNAseq2 = scRNAseq2.shape[1]
input_dim_scRNAseq2 = Input(shape = (ncol_scRNAseq2, ), name = "scRNAseq2")

encoding_dim_scRNAseq1 = 64
encoding_dim_scRNAseq2 = 64

dropout_scRNAseq1 = Dropout(0.1, name = "Dropout_scRNAseq1")(input_dim_scRNAseq1)
dropout_scRNAseq2 = Dropout(0.1, name = "Dropout_scRNAseq2")(input_dim_scRNAseq2)

encoded_scRNAseq1 = Dense(encoding_dim_scRNAseq1, activation = 'relu', name = "Encoder_scRNAseq1", use_bias=True, kernel_regularizer=WeightsOrthogonalityConstraint(64, weightage=1., axis=0))(dropout_scRNAseq1) #300 #prv 256 
encoded_scRNAseq2 = Dense(encoding_dim_scRNAseq2, activation = 'relu', name = "Encoder_scRNAseq2", use_bias=True, kernel_regularizer=WeightsOrthogonalityConstraint(64, weightage=1., axis=0))(dropout_scRNAseq2)

merge = concatenate([encoded_scRNAseq1,  encoded_scRNAseq2])

bottleneck = Dense(64, kernel_initializer = 'uniform', activation = 'linear', name = "Bottleneck")(merge) #50

merge_inverse = Dense(encoding_dim_scRNAseq1 + encoding_dim_scRNAseq2, activation = 'relu', name = "Concatenate_Inverse")(bottleneck)

decoded_scRNAseq1 = Dense(ncol_scRNAseq1, activation = 'relu', name = "Decoder_scRNAseq1")(merge_inverse) #sigmoid

decoded_scRNAseq2 = Dense(ncol_scRNAseq2, activation = 'relu', name = "Decoder_scRNAseq2")(merge_inverse)

autoencoder = Model([input_dim_scRNAseq1, input_dim_scRNAseq2],  [decoded_scRNAseq1, decoded_scRNAseq2])

opt = Adam(lr=0.0001)
autoencoder.compile(optimizer = opt, loss={'Decoder_scRNAseq1': 'mean_squared_error', 'Decoder_scRNAseq2': 'mean_squared_error'}) #loss_weights = [1., 1.]
autoencoder.summary()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=20)
# Autoencoder training
estimator = autoencoder.fit([scRNAseq1, scRNAseq2], [scRNAseq1, scRNAseq2], epochs = 600, batch_size = 32, validation_split = 0.2, shuffle = True, verbose = 1, callbacks=[es]) #prev 64 BS prev 32


encoder = Model([input_dim_scRNAseq1, input_dim_scRNAseq2], bottleneck)
bottleneck_representation = encoder.predict([scRNAseq1, scRNAseq2])

embd = pd.DataFrame(bottleneck_representation)
#embd  = scipy.sparse.csr_matrix(RNA_ATAC_Latent.values)

mod1_obs = ad_mod1.obs
mod1_uns = ad_mod1.uns
logging.info('Storing output to file')
adata = ad.AnnData(
    X=embd.values,
    obs=mod1_obs,
    uns={
        'dataset_id': mod1_uns['dataset_id'],
        'method_id': meta['functionality_name'],
    },
)
adata.write_h5ad(par['output'], compression="gzip")
