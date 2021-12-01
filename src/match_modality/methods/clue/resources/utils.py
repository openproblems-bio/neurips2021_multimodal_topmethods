r"""
Utilities for consistent data preprocessing
"""

from typing import Callable, Mapping, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd

import scglue


class Preprocessing:

    def __init__(self) -> None:
        self.params = {}

    def fit_transform(self, adata: ad.AnnData) -> None:
        raise NotImplementedError

    def transform(self, adata: ad.AnnData) -> None:
        raise NotImplementedError


class GEXPreprocessing(Preprocessing):

    r"""
    hvgs + scale + pca
    """

    MERGE_ADT = {
        "ADGRG1", "ANPEP", "B3GAT1", "BTLA", "C5AR1", "CCR2", "CCR4", "CCR5",
        "CCR6", "CD101", "CD109", "CD14", "CD151", "CD163", "CD19", "CD1C",
        "CD1D", "CD2", "CD200", "CD22", "CD226", "CD24", "CD244", "CD27",
        "CD274", "CD28", "CD33", "CD36", "CD37", "CD38", "CD3D", "CD4", "CD40",
        "CD40LG", "CD44", "CD47", "CD48", "CD5", "CD52", "CD55", "CD58", "CD63",
        "CD69", "CD7", "CD72", "CD74", "CD79B", "CD81", "CD82", "CD83", "CD84",
        "CD86", "CD8A", "CD9", "CD93", "CD99", "CLEC12A", "CLEC1B", "CLEC4C",
        "CR1", "CR2", "CSF1R", "CSF2RA", "CSF2RB", "CTLA4", "CX3CR1", "CXCR3",
        "CXCR5", "DPP4", "ENG", "ENTPD1", "F3", "FAS", "FCER1A", "FCER2",
        "FCGR1A", "FCGR2A", "FCGR3A", "GGT1", "GP1BB", "HLA-A", "HLA-DRA",
        "HLA-E", "ICAM1", "ICOS", "ICOSLG", "IFNGR1", "IGHD", "IGHE", "IGHM",
        "IGKC", "IL2RA", "IL2RB", "IL3RA", "IL4R", "IL7R", "ITGA1", "ITGA2",
        "ITGA2B", "ITGA4", "ITGA6", "ITGAE", "ITGAL", "ITGAM", "ITGAX", "ITGB1",
        "ITGB2", "ITGB3", "ITGB7", "KIR2DL1", "KIR2DL3", "KIR3DL1", "KLRB1",
        "KLRD1", "KLRF1", "KLRG1", "KLRK1", "LAG3", "LAIR1", "LAMP1", "LILRB1",
        "MCAM", "MS4A1", "NCAM1", "NCR1", "NECTIN2", "NRP1", "NT5E", "OLR1",
        "PDCD1", "PDPN", "PECAM1", "PTPRC", "PVR", "SELL", "SELP", "SELPLG",
        "SIGLEC1", "SIGLEC7", "SIRPA", "SLAMF1", "SLAMF6", "SLAMF7", "SPN",
        "TFRC", "THBD", "TIGIT", "TNFRSF13B", "TNFRSF13C", "TNFRSF14",
        "TNFRSF4", "TNFRSF9", "TRAV7", "TRDV2", "TREM1"
    }

    def __init__(
            self, n_comps: int = 100, n_genes: int = 2000,
            merge_adt: bool = False
    ) -> None:
        super().__init__()
        self.n_comps = n_comps
        self.n_genes = n_genes
        self.merge_adt = merge_adt

    def fit_transform(self, adata: ad.AnnData) -> None:
        sc.pp.highly_variable_genes(
            adata, layer="counts", n_top_genes=self.n_genes,
            flavor="seurat_v3", batch_key="batch"
        )
        if self.merge_adt:
            adata.var["highly_variable"] = [
                highly_variable or var_name in self.MERGE_ADT
                for var_name, highly_variable in
                zip(adata.var_names, adata.var["highly_variable"])
            ]
        features = adata.var.query("highly_variable").index.tolist()
        hvg = set(features)
        X = adata[:, features].X
        if scipy.sparse.issparse(X):
            mean = X.mean(axis=0).A1
            std = np.sqrt(X.power(2).mean(axis=0).A1 - np.square(mean))
            X = (X.toarray() - mean) / std
        else:
            mean = X.mean(axis=0)
            std = np.sqrt(X.square().mean(axis=0) - np.square(mean))
            X = (X - mean) / std
        X = X.clip(-10, 10)
        u, s, vh = randomized_svd(X.T @ X, self.n_comps, n_iter=15, random_state=0)
        adata.obsm["X_pca"] = X @ vh.T

        self.params["features"] = features
        self.params["hvg"] = hvg
        self.params["mean"] = mean
        self.params["std"] = std
        self.params["vh"] = vh

    def transform(self, adata: ad.AnnData) -> None:
        features = self.params["features"]
        hvg = self.params["hvg"]
        mean = self.params["mean"]
        std = self.params["std"]
        vh = self.params["vh"]

        adata.var["highly_variable"] = [i in hvg for i in adata.var_names]
        X = adata[:, features].X
        if scipy.sparse.issparse(X):
            X = (X.toarray() - mean) / std
        else:
            X = (X - mean) / std
        X = X.clip(-10, 10)
        adata.obsm["X_pca"] = X @ vh.T


class ATACPreprocessing(Preprocessing):

    r"""
    tfidf + normalize + log1p + svd + standardize
    """

    def __init__(self, n_comps: int = 100, n_peaks: int = 30000) -> None:
        super().__init__()
        self.n_comps = n_comps
        self.n_peaks = n_peaks

    def fit_transform(self, adata: ad.AnnData) -> None:
        top_idx = set(np.argsort(adata.X.sum(axis=0).A1)[-self.n_peaks:])
        adata.var["highly_variable"] = [i in top_idx for i in range(adata.n_vars)]
        features = adata.var_names.tolist()
        hvg = set(adata.var.query("highly_variable").index.tolist())
        X = adata[:, features].layers["counts"]
        idf = X.shape[0] / X.sum(axis=0).A1
        if scipy.sparse.issparse(X):
            tf = X.multiply(1 / X.sum(axis=1))
            X = tf.multiply(idf)
            X = X.multiply(1e4 / X.sum(axis=1))
        else:
            tf = X / X.sum(axis=1, keepdims=True)
            X = tf * idf
            X = X * (1e4 / X.sum(axis=1, keepdims=True))
        X = np.log1p(X)
        u, s, vh = randomized_svd(X, self.n_comps, n_iter=15, random_state=0)
        X_lsi = X @ vh.T / s
        X_lsi -= X_lsi.mean(axis=1, keepdims=True)
        X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
        adata.obsm["X_lsi"] = X_lsi

        self.params["features"] = features
        self.params["hvg"] = hvg
        self.params["idf"] = idf
        self.params["vh"] = vh
        self.params["s"] = s

    def transform(self, adata: ad.AnnData) -> None:
        features = self.params["features"]
        hvg = self.params["hvg"]
        idf = self.params["idf"]
        vh = self.params["vh"]
        s = self.params["s"]

        adata.var["highly_variable"] = [i in hvg for i in adata.var_names]
        X = adata[:, features].layers["counts"]
        if scipy.sparse.issparse(X):
            tf = X.multiply(1 / X.sum(axis=1))
            X = tf.multiply(idf)
            X = X.multiply(1e4 / X.sum(axis=1))
        else:
            tf = X / X.sum(axis=1, keepdims=True)
            X = tf * idf
            X = X * (1e4 / X.sum(axis=1))
        X = np.log1p(X)
        X_lsi = X @ vh.T / s
        X_lsi -= X_lsi.mean(axis=1, keepdims=True)
        X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
        adata.obsm["X_lsi"] = X_lsi


class ADTPreprocessing(Preprocessing):

    r"""
    scale + pca
    """

    def __init__(self, n_comps: int = 100) -> None:
        super().__init__()
        self.n_comps = n_comps

    def fit_transform(self, adata: ad.AnnData) -> None:
        adata.var["highly_variable"] = True
        features = adata.var_names.tolist()
        hvg = set(features)
        X = adata[:, features].X
        if scipy.sparse.issparse(X):
            mean = X.mean(axis=0).A1
            std = np.sqrt(X.power(2).mean(axis=0).A1 - np.square(mean))
            X = (X.toarray() - mean) / std
        else:
            mean = X.mean(axis=0)
            std = np.sqrt(X.square().mean(axis=0) - np.square(mean))
            X = (X - mean) / std
        X = X.clip(-10, 10)
        u, s, vh = randomized_svd(X.T @ X, self.n_comps, n_iter=15, random_state=0)
        adata.obsm["X_pca"] = X @ vh.T

        self.params["features"] = features
        self.params["hvg"] = hvg
        self.params["mean"] = mean
        self.params["std"] = std
        self.params["vh"] = vh

    def transform(self, adata: ad.AnnData) -> None:
        features = self.params["features"]
        hvg = self.params["hvg"]
        mean = self.params["mean"]
        std = self.params["std"]
        vh = self.params["vh"]

        adata.var["highly_variable"] = [i in hvg for i in adata.var_names]
        X = adata[:, features].X
        if scipy.sparse.issparse(X):
            X = (X.toarray() - mean) / std
        else:
            X = (X - mean) / std
        X = X.clip(-10, 10)
        adata.obsm["X_pca"] = X @ vh.T


def knn_matching(
        x: np.ndarray, y: np.ndarray, k: Optional[int] = None
) -> scipy.sparse.spmatrix:
    r"""
    K-nearest neighbor matching

    Parameters
    ----------
    x
        Representations of the first domain
    y
        Representations of the second domain
    k
        Number of neighbors to consider

    Returns
    -------
    matching_matrix
        Matching matrix
    """
    k = k or min(round(0.02 * y.shape[0]), 1000)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    nn = NearestNeighbors(n_neighbors=k).fit(y)
    nnd, nni = nn.kneighbors(x)
    nnd = nnd.astype(np.float64)
    ind_i = np.repeat(np.arange(x.shape[0]), k)
    ind_j = nni.flatten()
    sigma = nnd[:, :min(10, k)].mean(axis=1, keepdims = True) / 10
    ind_x = np.exp(-np.power(nnd / sigma, 2)).flatten()
    matching_matrix = scipy.sparse.csr_matrix(
        (ind_x, (ind_i, ind_j)),
        shape=(x.shape[0], y.shape[0])
    )
    matching_matrix = matching_matrix.multiply(1 / matching_matrix.sum(axis=1)).tocsr()
    return matching_matrix


def mnn_matching(
        x: np.ndarray, y: np.ndarray, k: Optional[int] = None
) -> scipy.sparse.spmatrix:
    r"""
    Mutual nearest neighbor matching

    Parameters
    ----------
    x
        Representations of the first domain
    y
        Representations of the second domain
    k
        Number of neighbors to consider

    Returns
    -------
    matching_matrix
        Matching matrix
    """
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)

    ky = k or min(round(0.01 * y.shape[0]), 1000)
    nny = NearestNeighbors(n_neighbors=ky).fit(y)
    x2y = nny.kneighbors_graph(x, mode="distance")
    x2y_binary = x2y.copy()
    x2y_binary.data[:] = 1.0

    kx = k or min(round(0.01 * x.shape[0]), 1000)
    nnx = NearestNeighbors(n_neighbors=kx).fit(x)
    y2x = nnx.kneighbors_graph(y, mode="distance").T  # Always x as rows
    y2x_binary = y2x.copy()
    y2x_binary.data[:] = 1.0

    mnn = x2y_binary.multiply(y2x_binary)
    matching_matrix = mnn.multiply(x2y).tolil()
    for i, data in enumerate(matching_matrix.data):
        if not len(data):
            continue
        data = np.array(data)
        sigma = data.mean() / 10
        matching_matrix.data[i] = np.exp(-np.power(data / sigma, 2)).tolist()
    matching_matrix = matching_matrix.tocsr()
    matching_matrix = matching_matrix.multiply(1 / matching_matrix.sum(axis=1)).tocsr()
    matching_matrix.data[~np.isfinite(matching_matrix.data)] = 0
    return matching_matrix


def snn_matching(
        x: np.ndarray, y: np.ndarray, k: Optional[int] = 1
) -> scipy.sparse.spmatrix:
    r"""
    Shared nearest neighbor matching

    Parameters
    ----------
    x
        Representations of the first domain
    y
        Representations of the second domain
    k
        Number of neighbors to consider

    Returns
    -------
    matching_matrix
        Matching matrix
    """
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)

    ky = k or min(round(0.01 * y.shape[0]), 1000)
    nny = NearestNeighbors(n_neighbors=ky).fit(y)
    x2y = nny.kneighbors_graph(x)
    y2y = nny.kneighbors_graph(y)

    kx = k or min(round(0.01 * x.shape[0]), 1000)
    nnx = NearestNeighbors(n_neighbors=kx).fit(x)
    y2x = nnx.kneighbors_graph(y)
    x2x = nnx.kneighbors_graph(x)

    x2y_intersection = x2y @ y2y.T
    y2x_intersection = y2x @ x2x.T
    jaccard = x2y_intersection + y2x_intersection.T
    jaccard.data = jaccard.data / (2 * kx + 2 * ky - jaccard.data)
    matching_matrix = jaccard.multiply(1 / jaccard.sum(axis=1)).tocsr()
    return matching_matrix


def ensemble_snn_matching(
        x: np.ndarray, y: np.ndarray, k: Optional[int] = 1
) -> scipy.sparse.spmatrix:
    r"""
    Ensemble shared nearest neighbor matching

    Parameters
    ----------
    x
        Ensemble representations of the first domain (n_sample * n_feature * n_ensemble)
    y
        Ensemble representations of the second domain (n_sample * n_feature * n_ensemble)
    k
        Number of neighbors to consider

    Returns
    -------
    matching_matrix
        Matching matrix
    """
    assert x.shape[1:] == y.shape[1:]
    n_ensemble = x.shape[2]
    x = np.moveaxis(x, 2, 0)  # (n_ensemble * n_sample * n_feature)
    y = np.moveaxis(y, 2, 0)  # (n_ensemble * n_sample * n_feature)
    x = x / np.linalg.norm(x, axis=2, keepdims=True)
    y = y / np.linalg.norm(y, axis=2, keepdims=True)

    ky = k or min(round(0.01 * y.shape[1]), 1000)
    kx = k or min(round(0.01 * x.shape[1]), 1000)

    x2y_intersection, y2x_intersection = [], []
    for x_, y_ in zip(x, y):
        nny = NearestNeighbors(n_neighbors=ky).fit(y_)
        x2y = nny.kneighbors_graph(x_)
        y2y = nny.kneighbors_graph(y_)

        nnx = NearestNeighbors(n_neighbors=kx).fit(x_)
        y2x = nnx.kneighbors_graph(y_)
        x2x = nnx.kneighbors_graph(x_)

        x2y_intersection.append(x2y @ y2y.T)
        y2x_intersection.append(y2x @ x2x.T)

    x2y_intersection = sum(x2y_intersection)
    y2x_intersection = sum(y2x_intersection)
    jaccard = x2y_intersection + y2x_intersection.T
    jaccard.data = jaccard.data / (2 * kx * n_ensemble + 2 * ky * n_ensemble - jaccard.data)
    matching_matrix = jaccard.multiply(1 / jaccard.sum(axis=1)).tocsr()
    return matching_matrix


def split_matching(
        mod1: ad.AnnData, mod2: ad.AnnData, keys: Tuple[str, str], split_by: str,
        models: Union[scglue.models.Model, Mapping[str, scglue.models.Model]],
        matching_function: Callable[[np.ndarray, np.ndarray], scipy.sparse.spmatrix],
        **kwargs
) -> scipy.sparse.spmatrix:
    r"""
    Splitted matching

    Parameters
    ----------
    mod1
        First modality data
    mod2
        Second modality data
    keys
        Modality keys
    split_by
        Split attribute (column in obs)
    models
        Model for each split.
        If a single model is provided, it is used for all splits.
    matching_function
        Matching function
    **kwargs
        Additional keyword arguments for passed to the matching function

    Returns
    -------
    matching_matrix
        Combined matching matrix
    """
    assert mod1.obs_names.is_unique
    assert mod2.obs_names.is_unique
    assert not mod1.obs[split_by].isna().any()
    assert not mod2.obs[split_by].isna().any()
    mod1_splits = set(mod1.obs[split_by])
    mod2_splits = set(mod2.obs[split_by])
    splits = mod1_splits | mod2_splits

    if isinstance(models, scglue.models.Model):
        models = {split: models for split in splits}
    matching_matrices, mod1_obs_names, mod2_obs_names = [], [], []
    for split in splits:
        mod1_split = mod1[mod1.obs[split_by] == split]
        mod2_split = mod2[mod2.obs[split_by] == split]
        mod1_obs_names.append(mod1_split.obs_names)
        mod2_obs_names.append(mod2_split.obs_names)
        model = models[split]
        matching_matrices.append(
            matching_function(
                model.encode_data(keys[0], mod1_split),
                model.encode_data(keys[1], mod2_split),
                **kwargs
            ) if mod1_split.n_obs and mod2_split.n_obs else
            scipy.sparse.csr_matrix((mod1_split.n_obs, mod2_split.n_obs))
        )
    mod1_obs_names = pd.Index(np.concatenate(mod1_obs_names))
    mod2_obs_names = pd.Index(np.concatenate(mod2_obs_names))
    combined_matrix = scipy.sparse.block_diag(matching_matrices, format="csr")
    return combined_matrix[
        mod1_obs_names.get_indexer(mod1.obs_names), :
    ][
        :, mod2_obs_names.get_indexer(mod2.obs_names)
    ]


def match_probabilities(
        pred: scipy.sparse.spmatrix, true: scipy.sparse.spmatrix
) -> np.ndarray:
    r"""
    Extract the matching probabilities

    Parameters
    ----------
    pred
        Predicted matching matrix
    true
        True matching matrix

    Returns
    ------
    probabilities
        Matching probabilities
        (keeping the same order as rows of the input matrices)

    Note
    ----
    To get the competition scores, use ``probabilities.mean() * 1000``.
    """
    true = true.tocoo()
    col_sort = true.col[np.argsort(true.row)]
    pred = pred.tocsc()[:, col_sort]
    return pred.diagonal()


def visualize_neighbors(
        mod1: ad.AnnData, mod2: ad.AnnData,
        joint_embedding: str, color: str,
        matching_matrix: scipy.sparse.spmatrix,
        linewidths: float = 0.1
) -> None:
    r"""
    Visualize the predicted matching between two modalities

    Parameters
    ----------
    mod1
        First domain
    mod2
        Second domain
    matching_matrix
        Matching matrix
    """
    combined = ad.AnnData(
        obs=pd.concat([mod1.obs, mod2.obs]),
        obsm={joint_embedding: np.concatenate([
            mod1.obsm[joint_embedding],
            mod2.obsm[joint_embedding]
        ])},
    )
    fig = sc.pl.embedding(combined, joint_embedding, color=color, return_fig=True)

    mod1, mod2 = mod1.obsm[joint_embedding], mod2.obsm[joint_embedding]
    matching_matrix = matching_matrix.tocoo()
    lines = [
        [mod1[row], mod2[col]]
        for row, col in zip(matching_matrix.row, matching_matrix.col)
    ]
    c = [(0, 0, 0, data) for data in matching_matrix.data]
    for ax in fig.axes:
        lc = mc.LineCollection(lines, colors=c, linewidths=linewidths, rasterized=True)
        ax.add_collection(lc)
    return fig


def get_cell_label(adata,cell_name =[]):
    r"""
    Get the cell label for cell classifier

    Parameters
    ----------
    adata
        adata with obs column 'cell_type'
    cell_name
        target cell_name list

    Returns
    ------
    y_label_int
        Bool list of if the cell is in the input cell labels
    """
    y_label = np.array(adata.obs['cell_type'].str.contains('|'.join(cell_name),case=True,regex=True))
    y_label_int = np.multiply(y_label,1)
    len(y_label_int)
    return y_label_int


def get_acc(outputs, labels) -> float:
    r"""
    Calculate the model predict accuracy

    Parameters
    ----------
    outputs
        Model output
    labels
        True label

    Returns
    ------
    acc
        Accuracy
    """
    _, predict = torch.max(outputs.data, 1)
    total_num = labels.shape[0]*1.0
    correct_num = (labels == predict).sum().item()
    acc = correct_num / total_num

    return acc


def train_cell_classifier(
    model,dataloader,batch_size,optim,range_num=20,cuda_index=7,
    loss_fun=torch.nn.CrossEntropyLoss()
) -> None:
    r"""
    Train cell type classifier

    Parameters
    ----------
    model
        Torch model in CUDA!
    dataloader
        Torch dataloader of train dataset
    range_num
        How many epoches you want
    batch_size
        batch size
    cuda_index
        which GPU to use(0-7 in 159)
    optim
        optimizer
    loss_fun
        loss function
    Returns
    ------
    None
    """
    # model = model.cuda(cuda_index)
    for e in range(range_num):
        epoch_loss = 0
        epoch_acc = 0
        for i, (x, y) in enumerate(dataloader):
            x = x.float()
            x = x.cuda(cuda_index)
            y = y.cuda(cuda_index)
            optim.zero_grad()

            out = model(x)
            loss = loss_fun(out, y)

            loss.backward()
            optim.step()

            epoch_loss += loss.data
            epoch_acc += get_acc(out, y)

        if e % 2 == 0:
            print('epoch: %d, loss: %f, acc: %f ' % (e, epoch_loss /5
            ,epoch_acc/int(len(dataloader.dataset))*batch_size))



def test_cell_classifier(model, dataloader, cuda_index=7) -> None:
    r"""
    Test the cell type classifier

    Parameters
    ----------
    model
        Torch model in CUDA!
    dataloader
        Torch dataloader of test dataset
    range_num
        How many epoches you want
    batch_size
        batch size
    dataset
        train dataset
    cuda_index
        which GPU to use(0-7 in 159), must same with train_cell_classifier()

    Returns
    ------
    None

    Note
    ----
        None
    """
    test_loss = 0
    correct = 0

    for data, target in dataloader:
        data = data.float()
        target = target.long()
        data = data.cuda(cuda_index)
        target = target.cuda(cuda_index)

        output = model(data)

        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(dataloader.dataset)
    print('\nTest set:, Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))


class ImbalancedDatasetSampler_fix(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        import torchvision  # FIXME: Is this really necessary? If so we need to add it as a dependency in viash config
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.tensors[1]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def get_confusion_matrix(trues, preds):
    r"""
    Get the confusion matrix

    Parameters
    ----------
    trues
        true label
    preds
        predict label by your model

    Returns
    ------
    confusion matrix
    """
    labels = range(len(set(trues)))
    conf_matrix = confusion_matrix(trues, preds, labels)
    return conf_matrix


def plot_confusion_matrix(conf_matrix,labels, fig_size=15, font_size=10):
    r"""
    Plot the confusion matrix

    Parameters
    ----------
    conf_matrix
        confusion matrix
    labels
        name of every class
    fig_size
        figure size of the picture
    font_size
        font size of the picture

    Returns
    ------
    None
    """
    plt.rcParams['figure.figsize'] = [fig_size, fig_size]
    plt.imshow(conf_matrix, cmap=plt.cm.Greens)
    indices = range(conf_matrix.shape[0])
    labels = range(len(labels))

    plt.xticks(indices, labels, rotation = 90)
    plt.yticks(indices, labels)
    plt.colorbar()
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    # show data
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(first_index, second_index, conf_matrix[first_index, second_index],size = font_size)
    # plt.savefig('gex_heatmap_confusion_matrix.jpg')
    plt.show()


def test_with_plot(model,test_dataloader, cuda_index = 7,fig_size=20):
    r"""
    Plot the confusion matrix

    Parameters
    ----------
    model
        trained model
    test_dataloader
        torch dataset loader of test data
    cuda_index
        GPU device index
    fig_size
        size of the picture

    Returns
    ------
    counfusion matrix
    """
    correct = 0
    test_preds = []
    test_trues = []
    for data, target in test_dataloader:
        data = data.float()
        target = target.long()
        data = data.cuda(cuda_index)
        target = target.cuda(cuda_index)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        test_preds.extend(pred.detach().cpu().numpy())
        test_trues.extend(target.detach().cpu().numpy())

    conf_matrix = get_confusion_matrix(test_trues, test_preds)
    print(conf_matrix)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))
    plt.figure(fig_size = (fig_size,fig_size))
    plot_confusion_matrix(conf_matrix)
    return conf_matrix
