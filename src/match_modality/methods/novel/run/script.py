#import pandas as pd
import anndata as ad
#import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from torch import nn
import numpy as np
import os
from itertools import chain
from typing import Callable, List, Mapping, Optional

import anndata
import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.preprocessing
import sklearn.neighbors
import sklearn.utils.extmath
import pickle
import sys

from scipy import sparse
from networkx.algorithms import bipartite

from scipy.sparse import csc_matrix

## VIASH START
dataset_path = "output/datasets/match_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_"
pretrain_path = "output/pretrain/match_modality/clue/openproblems_bmmc_cite_phase2_rna.clue_train.output_pretrain/"

par = {
    'input_train_mod1': f'{dataset_path}train_mod1.h5ad',
    'input_train_mod2': f'{dataset_path}train_mod2.h5ad',
    'input_train_sol': f'{dataset_path}train_sol.h5ad',
    'input_test_mod1': f'{dataset_path}test_mod1.h5ad',
    'input_test_mod2': f'{dataset_path}test_mod2.h5ad',
    'input_pretrain': pretrain_path,
    'output': 'output.h5ad'
}
meta = { 'resources_dir'='.',
       'functionality_name': '169594'}
## VIASH END


sys.path.append(meta['resources_dir'])

input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_train_sol = ad.read_h5ad(par['input_train_sol'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])
input_test_mod2 = ad.read_h5ad(par['input_test_mod2'])

def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    Parameters
    ----------
    X
        Input matrix
    Returns
    -------
    X_tfidf
        TF-IDF normalized matrix
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf

class tfidfTransformer():
    def __init__(self):
        self.idf = None
        self.fitted = False

    def fit(self, X):
        self.idf = X.shape[0] / X.sum(axis=0)
        self.fitted = True

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError('Transformer was not fitted on any data')
        if scipy.sparse.issparse(X):
            tf = X.multiply(1 / X.sum(axis=1))
            return tf.multiply(self.idf)
        else:
            tf = X / X.sum(axis=1, keepdims=True)
            return tf * self.idf

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
 

class lsiTransformer():
    def __init__(self,
                 n_components: int = 20,
                 drop_first=True,
                 use_highly_variable = None
                ):
        
        self.drop_first=drop_first
        self.n_components = n_components+drop_first
        self.use_highly_variable = use_highly_variable
        self.tfidfTransformer = tfidfTransformer()
        self.normalizer =  sklearn.preprocessing.Normalizer(norm="l1")
        self.pcaTransformer = sklearn.decomposition.TruncatedSVD(n_components = self.n_components, random_state=777)
        self.fitted = None
        
    def fit(self, adata: anndata.AnnData):
        if self.use_highly_variable is None:
            self.use_highly_variable = "highly_variable" in adata.var
        adata_use = adata[:, adata.var["highly_variable"]] if self.use_highly_variable else adata
        X = self.tfidfTransformer.fit_transform(adata_use.X)
        X_norm = self.normalizer.fit_transform(X)
        X_norm = np.log1p(X_norm * 1e4)
        X_lsi = self.pcaTransformer.fit_transform(X_norm)
        self.fitted = True
    
    def transform(self, adata):
        if not self.fitted:
            raise RuntimeError('Transformer was not fitted on any data')
        adata_use = adata[:, adata.var["highly_variable"]] if self.use_highly_variable else adata
        X = self.tfidfTransformer.transform(adata_use.X)
        X_norm = self.normalizer.transform(X)
        X_norm = np.log1p(X_norm * 1e4)
        X_lsi = self.pcaTransformer.transform(X_norm)
        X_lsi -= X_lsi.mean(axis=1, keepdims=True)
        X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
        lsi_df = pd.DataFrame(X_lsi, index = adata_use.obs_names).iloc[:,int(self.drop_first):]
        return lsi_df
    
    def fit_transform(self, adata):
        self.fit(adata)
        return self.transform(adata)
        
        
                 
    
def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    use_highly_variable
        Whether to use highly variable features only, stored in
        ``adata.var['highly_variable']``. By default uses them if they
        have been determined beforehand.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, random_state=777, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi



class ModalityMatchingDataset(Dataset):
    def __init__(
        self, df_modality1, df_modality2, df_adjacency_matrix, is_train
    ):
        super().__init__()
        self.is_train = is_train
        
        self.df_modality1 = df_modality1
        self.df_modality2 = df_modality2
        self.df_adjacency_matrix = df_adjacency_matrix
    
    def __len__(self):
        return self.df_modality1.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if(self.is_train):
            x_modality_1 = self.df_modality1.iloc[index].values
            x_modality_2 = self.df_modality2.iloc[self.df_adjacency_matrix.iloc[index].argmax()].values
        else:
            x_modality_1 = self.df_modality1.iloc[index].values
            x_modality_2 = self.df_modality2.iloc[index].values
        
        return x_modality_1, x_modality_2
    
class Encoder(nn.Module):
    def __init__(self, n_input, embedding_size, dropout_rates, dims_layers):
        super(Encoder, self).__init__()
        dropout = []
        layers = []
        layers.append(nn.Linear(n_input, dims_layers[0]))
        
        for i in range(len(dims_layers)-1):
            layers.append(nn.Linear(dims_layers[i], dims_layers[i+1]))
        for i in range(len(dropout_rates)):
            dropout.append(nn.Dropout(p=dropout_rates[i]))
            
        layers.append(nn.Linear(dims_layers[-1], embedding_size))
        
        self.fc_list = nn.ModuleList(layers)
        self.dropout_list = nn.ModuleList(dropout)
        
    def forward(self, x):
        for i in range(len(self.fc_list)-1):
            x = F.elu(self.fc_list[i](x))
            if(i<len(self.dropout_list)):
                 x = self.dropout_list[i](x)
            
        x = self.fc_list[-1](x)
        return x

class Modality_CLIP(nn.Module):
    def __init__(self, Encoder, layers_dims, dropout_rates, dim_mod1, dim_mod2, output_dim, T):
        super(Modality_CLIP, self).__init__()
        
        self.encoder_modality1 = Encoder(dim_mod1, output_dim, dropout_rates[0], layers_dims[0])
        self.encoder_modality2 = Encoder(dim_mod2, output_dim, dropout_rates[1], layers_dims[1])
        
        self.logit_scale = nn.Parameter(torch.ones([]) * T)
        
    def forward(self, features_first, features_second):
        features_mod1 = self.encoder_modality1(features_first)
        features_mod2 = self.encoder_modality2(features_second)
        
        features_mod1 = features_mod1/torch.norm(features_mod1, p=2, dim=-1, keepdim=True)
        features_mod2 = features_mod2/torch.norm(features_mod2, p=2, dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        
        logits = logit_scale*features_mod1@features_mod2.T
        
        return logits, features_mod1, features_mod2
def get_bipartite_matching_adjacency_matrix_mk3(raw_logits, threshold_quantile=0.995, copy=False):
    #getting rid of unpromising graph connections
    if copy:
        weights = raw_logits.copy()
    else:
        weights = raw_logits
    quantile_row = np.quantile(weights, threshold_quantile, axis=0, keepdims=True)
    quantile_col = np.quantile(weights, threshold_quantile, axis=1, keepdims=True)
    #quantile_minimum = np.minimum(quantile_row, quantile_col, out=quantile_row)
    mask_ = (weights<quantile_row)
    mask_ = np.logical_and(mask_, (weights<quantile_col), out=mask_)
    #weights[weights<quantile_minimum] = 0
    weights[mask_] = 0
    weights_sparse = sparse.csr_matrix(-weights)
    del(weights)
    graph = bipartite.matrix.from_biadjacency_matrix(weights_sparse)
    #explicitly combining top nodes in once component or networkx freaks tf out
    u = [n for n in graph.nodes if graph.nodes[n]['bipartite'] == 0]
    matches = bipartite.matching.minimum_weight_full_matching(graph, top_nodes=u)
    best_matches = np.array([matches[x]-len(u) for x in u])
    bipartite_matching_adjacency = np.zeros(raw_logits.shape)
    bipartite_matching_adjacency[np.arange(raw_logits.shape[0]), best_matches]=1
    return bipartite_matching_adjacency

    
if(input_test_mod1.to_df().shape[1] == 134 or input_test_mod2.to_df().shape[1]==134):
    weight = torch.load(par['input_pretrain'] + '/best.pth', map_location='cpu')
        
        
    model = Modality_CLIP(Encoder, 
                         ([512], 
                          [512,2048,1024,2048]), 
                         ([0.199296], 
                          [0.03524, 0.531454, 0.254134, 0.203471]),
                          134,
                          256,
                          64, 
                          2.739896)
    
    with open(par['input_pretrain'] + '/lsi_transformer.pickle', 'rb') as f:
        lsi_transformer_gex = pickle.load(f)
    if(input_test_mod1.to_df().shape[1] == 134):
        input_test_mod2 = lsi_transformer_gex.transform(input_test_mod2)
        input_test_mod1 = input_test_mod1.to_df()
    else:
        input_test_mod1 = lsi_transformer_gex.transform(input_test_mod1)
        input_test_mod2 = input_test_mod2.to_df()
        
        
    
    model.load_state_dict(weight['model_state_dict'])
    dataset_test = ModalityMatchingDataset(input_test_mod1, input_test_mod2, None, is_train = False)
    data_test = torch.utils.data.DataLoader(dataset_test, 32, shuffle = False)
    
    all_emb_mod1 = []
    all_emb_mod2 = []
    indexes = []
    model.eval();
    for x1, x2 in data_test:
        if(x1.shape[1] == 134):
            logits,features_mod1, features_mod2 = model(x1, x2)
        elif(x1.shape[1] == 256):
            logits, features_mod1, features_mod2 = model(x2, x1)
        
        all_emb_mod1.append(features_mod1.detach().cpu())
        all_emb_mod2.append(features_mod2.detach().cpu())
        
    all_emb_mod1 = torch.cat(all_emb_mod1)
    all_emb_mod2 = torch.cat(all_emb_mod2)
    if(x1.shape[1] == 134):
        out1_2 = (all_emb_mod1@all_emb_mod2.T).detach().cpu().numpy()
    elif(x1.shape[1] == 256):
        out1_2 = ((all_emb_mod1@all_emb_mod2.T).T).detach().cpu().numpy()
        
    out1_2 = get_bipartite_matching_adjacency_matrix_mk3(out1_2, threshold_quantile=0.990)
    out1_2 = csc_matrix(out1_2)
    out = ad.AnnData(
        X=out1_2,
        uns={
            "dataset_id": input_train_mod1.uns["dataset_id"],
            "method_id": meta['functionality_name']
        }
    )
    out.write_h5ad(par['output'], compression="gzip")
    
else:
    shape_1 = input_test_mod1.shape[1]
    weight = torch.load(par['input_pretrain'] + '/best.pth', map_location='cpu')      
    model = Modality_CLIP(Encoder, 
                         ([2048], 
                          [1024, 1024]), 
                         ([0.661497], 
                          [0.541996, 0.396641]),
                          512,
                          64,
                          256, 
                          3.065016)
    
    with open(par['input_pretrain'] + '/lsi_GEX_transformer.pickle', 'rb') as f:
        lsi_transformer_gex = pickle.load(f)
    with open(par['input_pretrain'] + '/lsi_ATAC_transformer.pickle', 'rb') as f:
        lsi_transformer_atac = pickle.load(f)
    if(input_test_mod1.to_df().shape[1] == 116490):
        input_test_mod2 = lsi_transformer_gex.transform(input_test_mod2)
        input_test_mod1 = lsi_transformer_atac.transform(input_test_mod1)
    else:
        input_test_mod1 = lsi_transformer_gex.transform(input_test_mod1)
        input_test_mod2 = lsi_transformer_atac.transform(input_test_mod2)
    
    model.load_state_dict(weight['model_state_dict'])
    
    dataset_test = ModalityMatchingDataset(input_test_mod1, input_test_mod2, None, is_train = False)
    data_test = torch.utils.data.DataLoader(dataset_test, 32, shuffle = False)
    
    all_emb_mod1 = []
    all_emb_mod2 = []
    indexes = []
    model.eval();
    for x1, x2 in data_test:
        if(shape_1 == 116490):
            logits,features_mod1, features_mod2 = model(x1, x2)
        else:
            logits, features_mod1, features_mod2 = model(x2, x1)
            
        all_emb_mod1.append(features_mod1.detach().cpu())
        all_emb_mod2.append(features_mod2.detach().cpu())
        
    all_emb_mod1 = torch.cat(all_emb_mod1)
    all_emb_mod2 = torch.cat(all_emb_mod2)
    if(shape_1 == 116490):
        out1_2 = (all_emb_mod1@all_emb_mod2.T).detach().cpu().numpy()
    else:
        out1_2 = ((all_emb_mod1@all_emb_mod2.T).T).detach().cpu().numpy()
        
    out1_2 = get_bipartite_matching_adjacency_matrix_mk3(out1_2, threshold_quantile=0.990)
    out1_2 = csc_matrix(out1_2)
    out = ad.AnnData(
        X=out1_2,
        uns={
            "dataset_id": input_train_mod1.uns["dataset_id"],
            "method_id": meta['functionality_name']
        }
    )
    out.write_h5ad(par['output'], compression="gzip")
        
        
    