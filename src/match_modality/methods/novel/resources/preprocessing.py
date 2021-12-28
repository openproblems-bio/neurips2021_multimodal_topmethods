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
        # self.lsi_mean = None
        # self.lsi_std = None
        self.fitted = None
        
    def fit(self, adata: anndata.AnnData):
        if self.use_highly_variable is None:
            self.use_highly_variable = "highly_variable" in adata.var
        adata_use = adata[:, adata.var["highly_variable"]] if self.use_highly_variable else adata
        X = self.tfidfTransformer.fit_transform(adata_use.X)
        X_norm = self.normalizer.fit_transform(X)
        X_norm = np.log1p(X_norm * 1e4)
        X_lsi = self.pcaTransformer.fit_transform(X_norm)
        # self.lsi_mean = X_lsi.mean(axis=1, keepdims=True)
        # self.lsi_std = X_lsi.std(axis=1, ddof=1, keepdims=True)
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