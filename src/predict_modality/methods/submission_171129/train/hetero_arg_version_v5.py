# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
import math
import json
import re
import argparse
import pickle
import anndata as ad

import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

import dgl
from dgl.heterograph import DGLBlock
from dgl import function as fn
from dgl.transform import reverse
import dgl.nn as dglnn


parser = argparse.ArgumentParser()
parser.add_argument('prefix')
parser.add_argument('-t', '--subtask', default = 'openproblems_bmmc_cite_phase2_rna')
parser.add_argument('-d', '--data_folder', default = './data/public/phase2-data/predict_modality/')
parser.add_argument('-pww', '--pathway_weight', default = 'cos', choices = ['cos', 'one', 'pearson'])
parser.add_argument('-pwth', '--pathway_threshold', type=float, default = 0.0)
parser.add_argument('-l', '--log_folder', default = './logs')
parser.add_argument('-m', '--model_folder', default = './models')
parser.add_argument('-r', '--result_folder', default = './results')
parser.add_argument('-e', '--epoch', type=int, default = 1500)
parser.add_argument('-nbf', '--no_batch_features', action = 'store_true')
parser.add_argument('-bf', '--batch_features', type=str, default='cell_summary') 
parser.add_argument('-npw', '--no_pathway', action = 'store_true')
parser.add_argument('-opw', '--only_pathway', action = 'store_true')
parser.add_argument('-res', '--residual', default = 'res_cat', choices=['none', 'res_add', 'res_cat'])
parser.add_argument('-inres', '--initial_residual', action = 'store_true')
parser.add_argument('-pwagg', '--pathway_aggregation', default = 'alpha', choices=['sum', 'attention', 'two_gate', 'one_gate', 'alpha', 'cat'])
parser.add_argument('-pwalpha', '--pathway_alpha', type=float, default = 0.5)
parser.add_argument('-nrc', '--no_readout_concatenate', action = 'store_true')
parser.add_argument('-bs', '--batch_size', default = 1000, type = int)
parser.add_argument('-nm', '--normalization', default = 'group', choices = ['batch', 'layer', 'group', 'none'])
parser.add_argument('-ac', '--activation', default = 'leaky_relu', choices = ['leaky_relu', 'relu', 'prelu', 'gelu']) 
parser.add_argument('-em', '--embedding_layers', default = 2, type=int, choices = [1,2,3])
parser.add_argument('-ro', '--readout_layers', default = 1, type=int, choices = [1,2])
parser.add_argument('-conv', '--conv_layers', default = 2, type=int, choices = [1,2,3,4,5,6])
parser.add_argument('-agg', '--agg_function', default = 'gcn', choices = ['gcn', 'mean'])
parser.add_argument('-device', '--device', default = 'cuda')
parser.add_argument('-sb', '--save_best', action = 'store_true')
parser.add_argument('-sf', '--save_final', action = 'store_true')
parser.add_argument('-lr', '--learning_rate', type=float, default = 1e-2)
parser.add_argument('-wd', '--weight_decay', type=float, default = 1e-5)
parser.add_argument('-hid', '--hidden_size', type=int, default = 48)
parser.add_argument('-edd', '--edge_dropout', type=float, default = 0.4)
parser.add_argument('-mdd', '--model_dropout', type=float, default = 0.3)
parser.add_argument('-es', '--early_stopping', type=int, default = 0)
parser.add_argument('-c', '--cpu', type=int, default = 8)
parser.add_argument('-ov', '--overlap', action = 'store_true')
parser.add_argument('-or', '--output_relu', default = 'none', choices = ['relu', 'leaky_relu', 'none'])
parser.add_argument('-i', '--inductive', default = 'trans', choices = ['normal', 'opt', 'trans'])
parser.add_argument('-sa', '--subpath_activation', action = 'store_true')
parser.add_argument('-ci', '--cell_init', default = 'none', choices=['none', 'ae'])
parser.add_argument('-bas', '--batch_seperation', action = 'store_true')

args = parser.parse_args()
PREFIX = args.prefix
logger = open(f'{args.log_folder}/{PREFIX}.log', 'w')
logger.write(str(args)+'\n')

torch.set_num_threads(args.cpu)
subtask = args.subtask
subtask_folder = args.data_folder + subtask + '/'
subtask_filename = subtask_folder + subtask + '.censor_dataset.output_{}.h5ad'

train_mod1 = ad.read_h5ad(subtask_filename.format('train_mod1'))
train_mod2 = ad.read_h5ad(subtask_filename.format('train_mod2'))

if args.batch_seperation:
    mask = pickle.load(open('phase2_mask_sep.pkl', 'rb'))[subtask]
else:
    mask = pickle.load(open('phase2_mask.pkl', 'rb'))[subtask]

# This will get passed to the method
input_train_mod1 = train_mod1.X[mask['train']]
input_train_mod2 = train_mod2.X[mask['train']]
input_test_mod1 =  train_mod1.X[mask['test']]
true_test_mod2 =  train_mod2.X[mask['test']]

FEATURE_SIZE = input_train_mod1.shape[1]
CELL_SIZE = input_train_mod1.shape[0] + input_test_mod1.shape[0]
OUTPUT_SIZE = input_train_mod2.shape[1]
TRAIN_SIZE = input_train_mod1.shape[0]
TEST_SIZE = input_test_mod1.shape[0]

class Conv1dEncoder2(nn.Module):
    
    def __init__(self,input_feature,latent_dim):
        super().__init__()
        self.relu2 = nn.ReLU()
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(input_feature, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
            
            # nn.Linear(input_feature//4, input_feature//16),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
        )
    def forward(self,x):
        x = self.encoder(x)
        return x  

class Decoder2(nn.Module):
    def __init__(self,latent_dim,output_feature):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),

            # nn.Linear(input_feature//16, input_feature//4),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(latent_dim, output_feature),
            # nn.BatchNorm1d(output_feature),
            nn.ReLU()
        )

    def forward(self,x):
        return self.decoder(x)

class AutoEncoder2En1De(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim):
        super().__init__()
        output_dim1, output_dim2 = input_dim1, input_dim2
        self.encoder1 = Conv1dEncoder2(input_dim1, hidden_dim)
        self.encoder2 = Conv1dEncoder2(input_dim2, hidden_dim)
        self.decoder2 = Decoder2(hidden_dim, output_dim2)
    
    def translate_1_to_2(self, x):
        e1 = self.encoder1(x)
        d2 = self.decoder2(e1)
        return d2
    
    def get_embedding(self, x):
        h = self.encoder1(x)
        return h

    def forward(self, x):
        input_modality1 = x[0]
        input_modality2 = x[1]
        latent_embed_modality1 = self.encoder1(input_modality1)
        latent_embed_modality2 = self.encoder2(input_modality2)
        output_modality2_transform = self.decoder2(latent_embed_modality1)
        output_modality2_itself = self.decoder2(latent_embed_modality2)
        return output_modality2_transform, output_modality2_itself, latent_embed_modality1 - latent_embed_modality2
    
class udfEdgeWeightNorm(nn.Module):
    def __init__(self, norm='both', eps=0.):
        super(udfEdgeWeightNorm, self).__init__()
        self._norm = norm
        self._eps = eps

    def forward(self, graph, edge_weight):
        with graph.local_scope():
            if isinstance(graph, DGLBlock):
                graph = block_to_graph(graph)
            if len(edge_weight.shape) > 1:
                raise DGLError('Currently the normalization is only defined '
                               'on scalar edge weight. Please customize the '
                               'normalization for your high-dimensional weights.')
            if self._norm == 'both' and th.any(edge_weight <= 0).item():
                raise DGLError('Non-positive edge weight detected with `norm="both"`. '
                               'This leads to square root of zero or negative values.')

            dev = graph.device
            graph.srcdata['_src_out_w'] = torch.ones((graph.number_of_src_nodes())).float().to(dev)
            graph.dstdata['_dst_in_w'] = torch.ones((graph.number_of_dst_nodes())).float().to(dev)
            graph.edata['_edge_w'] = edge_weight

            if self._norm in ['both', 'column', 'left']:
                reversed_g = reverse(graph)
                reversed_g.edata['_edge_w'] = edge_weight
                reversed_g.update_all(fn.copy_edge('_edge_w', 'm'), fn.sum('m', 'out_weight'))
                degs = reversed_g.dstdata['out_weight'] + self._eps
                norm = torch.pow(degs, -0.5)
                graph.srcdata['_src_out_w'] = norm

            if self._norm in ['both', 'row', 'right']:
                graph.update_all(fn.copy_edge('_edge_w', 'm'), fn.sum('m', 'in_weight'))
                degs = graph.dstdata['in_weight'] + self._eps
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                graph.dstdata['_dst_in_w'] = norm

            graph.apply_edges(lambda e: {'_norm_edge_weights': e.src['_src_out_w'] * \
                                                               e.dst['_dst_in_w'] * \
                                                               e.data['_edge_w']})
            return graph.edata['_norm_edge_weights']

# Pathway Features
import pickle
from collections import defaultdict
pww = args.pathway_weight
npw = args.no_pathway
uu=[]
vv=[]
ee=[]
if npw:
    pass
elif pww == 'cos' and subtask == 'openproblems_bmmc_cite_phase2_rna':
    uu, vv, ee = pickle.load(open('pw.pkl', 'rb'))
    ee = [e.item() for e in ee]
elif pww == 'cos' and subtask == 'openproblems_bmmc_multiome_phase2_rna':
    uu, vv, ee = pickle.load(open('pw_multiome.pkl', 'rb'))
    ee = [e.item() for e in ee]
else:
    with open('./h.all.v7.4.entrez.gmt') as gmt:
        gene_list = gmt.read().split()

    gene_sets_entrez = defaultdict(list)
    indicator = 0
    for ele in gene_list:
        if not ele.isnumeric() and indicator == 1:
            indicator = 0
            continue
        if not ele.isnumeric() and indicator == 0:
            indicator = 1
            gene_set_name = ele
        else:
            gene_sets_entrez[gene_set_name].append(ele)

    with open('./h.all.v7.4.symbols.gmt') as gmt:
        gene_list = gmt.read().split()

    gene_sets_symbols = defaultdict(list)

    for ele in gene_list:
        if ele in gene_sets_entrez:
            gene_set_name = ele
        elif not ele.startswith( 'http://' ):
            gene_sets_symbols[gene_set_name].append(ele)

    pw = [i[1] for i in gene_sets_symbols.items()]

    counter = 0
    total = 0
    feature_index = train_mod1.var['feature_types'].index.tolist()
    new_pw = []
    for i in pw:
        new_pw.append([])
        for j in i:
            if j in feature_index:
                new_pw[-1].append(feature_index.index(j))

    if pww == 'cos':
        for i in new_pw:
            for j in i:
                for k in i:
                    if j!=k:
                        uu.append(j)
                        vv.append(k)
                        sj = np.sqrt(np.dot(input_train_mod1[:,j].toarray().T, input_train_mod1[:,j].toarray()).item())
                        sk = np.sqrt(np.dot(input_train_mod1[:,k].toarray().T, input_train_mod1[:,k].toarray()).item())
                        jk = np.dot(input_train_mod1[:,j].toarray().T, input_train_mod1[:,k].toarray())
                        cossim = jk/sj/sk
                        ee.append(cossim.item())
    elif pww == 'one':
        for i in new_pw:
            for j in i:
                for k in i:
                    if j!=k:
                        uu.append(j)
                        vv.append(k)
                        ee.append(1)
    elif pww == 'pearson':
        corr = np.corrcoef(input_train_mod1.toarray().T)
        for i in new_pw:
            for j in i:
                for k in i:
                    if j!=k:
                        uu.append(j)
                        vv.append(k)
                        ee.append(corr[j][k])
                        
pwth = args.pathway_threshold
if pwth>0:
    nu = []
    nv = []
    ne = []

    for i in range(len(uu)):
        if ee[i] > pwth:
            ne.append(ee[i])
            nu.append(uu[i])
            nv.append(vv[i])
    uu, vv, ee = nu, nv, ne

# Batch Features
if args.no_batch_features:
    batch_features = None
else:
    cells = []
    columns = ['cell_mean', 'cell_std', 'nonzero_25%', 'nonzero_50%', 'nonzero_75%', 'nonzero_max', 'nonzero_count', 'nonzero_mean', 'nonzero_std', 'batch']

    bcl = list(train_mod1.obs['batch'][mask['train']])
    for i, cell in enumerate(input_train_mod1):
        cell = cell.toarray()
        nz = cell[np.nonzero(cell)]
        cells.append([cell.mean(), cell.std(), np.percentile(nz, 25), np.percentile(nz, 50), np.percentile(nz, 75), cell.max(), len(nz)/1000, nz.mean(), nz.std(), bcl[i]])

    bcl = list(train_mod1.obs['batch'][mask['test']])
    for i, cell in enumerate(input_test_mod1):
        cell = cell.toarray()
        nz = cell[np.nonzero(cell)]
        cells.append([cell.mean(), cell.std(), np.percentile(nz, 25), np.percentile(nz, 50), np.percentile(nz, 75), cell.max(), len(nz)/1000, nz.mean(), nz.std(), bcl[i]])

    cell_features = pd.DataFrame(cells, columns=columns)
    batch_source = cell_features.groupby('batch').mean().reset_index()
    batch_list = batch_source.batch.tolist()
    batch_source = batch_source.drop('batch', axis=1).to_numpy().tolist()
    b2i = dict(zip(batch_list, range(len(batch_list))))
    batch_features = []

    for b in train_mod1.obs['batch'][mask['train']]:
        batch_features.append(batch_source[b2i[b]])

    for b in train_mod1.obs['batch'][mask['test']]:
        batch_features.append(batch_source[b2i[b]])

    batch_features = torch.tensor(batch_features).float()

# Graph construction
if args.cell_init == 'none':
    cell_ids = torch.ones(CELL_SIZE).long()
else:
    model = AutoEncoder2En1De(FEATURE_SIZE, OUTPUT_SIZE, 100)
    model.load_state_dict(torch.load(f'ensemble_models/{subtask}_auto_encoder_model.pth', map_location='cpu'))
    model.eval()
    with torch.no_grad():
        cell_ids = torch.cat([model.get_embedding(torch.from_numpy(input_train_mod1.toarray())).detach(), model.get_embedding(torch.from_numpy(input_test_mod1.toarray())).detach()], 0).float()
    
def graph_construction(u, v, e, test = False):
    
    if args.only_pathway:
        graph_data = {
            ('feature', 'entail', 'cell'): (v, u),
            ('feature', 'pathway', 'feature'): (uu, vv),
        }
        graph = dgl.heterograph(graph_data)
        
        
        if args.inductive != 'trans':
            graph.nodes['cell'].data['id'] = cell_ids[:TRAIN_SIZE] if not test else cell_ids
        else:
            graph.nodes['cell'].data['id'] = cell_ids
            
        graph.nodes['feature'].data['id'] = torch.arange(FEATURE_SIZE).long()
        graph.edges['entail'].data['weight'] = e
        graph.edges['pathway'].data['weight'] = torch.tensor(ee).float()

    elif args.no_pathway:
        if args.inductive == 'opt':
            graph_data = {
                ('cell', 'occur', 'feature'): (u, v) if not test else (u[:g.edges(etype='occur')[0].shape[0]], v[:g.edges(etype='occur')[0].shape[0]]),
                ('feature', 'entail', 'cell'): (v, u),
            }
        
        else:
            graph_data = {
                ('cell', 'occur', 'feature'): (u, v),
                ('feature', 'entail', 'cell'): (v, u),
            }

        graph = dgl.heterograph(graph_data)
        
        if args.inductive != 'trans':
            graph.nodes['cell'].data['id'] = cell_ids[:TRAIN_SIZE] if not test else cell_ids
        else:
            graph.nodes['cell'].data['id'] = cell_ids
        graph.nodes['feature'].data['id'] = torch.arange(FEATURE_SIZE).long()
        graph.edges['entail'].data['weight'] = e
        graph.edges['occur'].data['weight'] = e[:graph.edges(etype='occur')[0].shape[0]]

    else:
        if args.inductive == 'opt':
            graph_data = {
                ('cell', 'occur', 'feature'): (u, v) if not test else (u[:g.edges(etype='occur')[0].shape[0]], v[:g.edges(etype='occur')[0].shape[0]]),
                ('feature', 'entail', 'cell'): (v, u),
                ('feature', 'pathway', 'feature'): (uu, vv),
            }
        else:
            graph_data = {
                ('cell', 'occur', 'feature'): (u, v),
                ('feature', 'entail', 'cell'): (v, u),
                ('feature', 'pathway', 'feature'): (uu, vv),
            }
        graph = dgl.heterograph(graph_data)

        if args.inductive != 'trans':
            graph.nodes['cell'].data['id'] = cell_ids[:TRAIN_SIZE] if not test else cell_ids
        else:
            graph.nodes['cell'].data['id'] = cell_ids
        graph.nodes['feature'].data['id'] = torch.arange(FEATURE_SIZE).long()
        graph.edges['entail'].data['weight'] = e
        graph.edges['occur'].data['weight'] = e[:graph.edges(etype='occur')[0].shape[0]]
        graph.edges['pathway'].data['weight'] = torch.tensor(ee).float()
        
    return graph

if args.inductive != 'trans':
    u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0]+i) for i, t in enumerate(input_train_mod1)], axis=0))
    v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1], axis=0))
    e = torch.from_numpy(input_train_mod1.tocsr().data).float()
    g = graph_construction(u, v, e)
    
    u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0]+i) for i, t in enumerate(input_train_mod1)] +\
                       [np.array(t.nonzero()[0]+i+TRAIN_SIZE) for i, t in enumerate(input_test_mod1)], axis=0))
    v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1] +\
                       [np.array(t.nonzero()[1]) for t in input_test_mod1], axis=0))
    e = torch.from_numpy(np.concatenate([input_train_mod1.tocsr().data, input_test_mod1.tocsr().data], axis=0)).float()
    gtest = graph_construction(u, v, e, test=True)
    
else:
    u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0]+i) for i, t in enumerate(input_train_mod1)] +\
                       [np.array(t.nonzero()[0]+i+TRAIN_SIZE) for i, t in enumerate(input_test_mod1)], axis=0))
    v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1] +\
                       [np.array(t.nonzero()[1]) for t in input_test_mod1], axis=0))
    e = torch.from_numpy(np.concatenate([input_train_mod1.tocsr().data, input_test_mod1.tocsr().data], axis=0)).float()
    g = graph_construction(u, v, e)
    
    gtest = g

if args.overlap:
    f = open('ADT_GEX.txt', 'r')
    gex_list = []
    for line in f:
        rs = re.search(':  (.*) contained in both Protein and RNA gene dataset', line)
        if rs is not None:
            gex_list.append(rs[1])

        rs = re.search('but not in RNA gene dataset \((.*)\)', line)
        if rs is not None and rs[1].find('not exist')==-1 and rs[1]!='HLA-A,HLA-B,HLA-C' and rs[1]!='CD1a':
            gex_list.append(rs[1])

    gex_list += ['HLA-A','HLA-B','HLA-C']
    gex_feature = torch.zeros(FEATURE_SIZE, 2)
    feature_list = list(train_mod1.var['feature_types'].index)
    for gex in gex_list:
        if gex in feature_list:
            ind = feature_list.index(gex)
            gex_feature[ind][0] = 1

    gex_list = pd.read_csv('ATAC_GEX_Overlap.csv').overlap_GEX.unique().tolist()
    for gex in gex_list:
        if gex in feature_list:
            gex_feature[feature_list.index(gex)][1] = 1

# data loader
train_labels = torch.from_numpy(input_train_mod2.toarray())
test_labels = torch.from_numpy(true_test_mod2.toarray())
BATCH_SIZE = args.batch_size
    
def validate(model):
    model.eval()
    with torch.no_grad():
        logits = model(gtest)
        logits = logits[-TEST_SIZE:]
        labels = test_labels
        loss = math.sqrt(F.mse_loss(logits, labels).item())
        logger.write(f'validation loss:  {loss}\n')
        logger.flush()
        return loss
    
# model
class WeightedGCN4(nn.Module):
    def __init__(self, hid_feats, out_feats):
        super().__init__()
        self.opw = args.only_pathway
        self.npw = args.no_pathway
        self.nrc = args.no_readout_concatenate
        
        if batch_features is not None:
            self.extra_encoder = nn.Linear(batch_features.shape[1], hid_feats)
        if args.overlap:
            self.ov_encoder = nn.Linear(2, hid_feats)
            
        if args.cell_init == 'none':
            self.embed_cell = nn.Embedding(2, hid_feats)
        else:
            self.embed_cell = nn.Linear(100, hid_feats)
            
        self.embed_feat = nn.Embedding(FEATURE_SIZE, hid_feats)
        
        self.input_linears = nn.ModuleList()
        self.input_acts = nn.ModuleList()
        self.input_norm = nn.ModuleList()
        for i in range((args.embedding_layers-1)*2):
            self.input_linears.append(nn.Linear(hid_feats, hid_feats))
        if args.activation == 'gelu':
            for i in range((args.embedding_layers-1)*2):
                self.input_acts.append(nn.GELU())
        elif args.activation == 'prelu':
            for i in range((args.embedding_layers-1)*2):
                self.input_acts.append(nn.PReLU())
        elif args.activation == 'relu':
            for i in range((args.embedding_layers-1)*2):
                self.input_acts.append(nn.ReLU())
        elif args.activation == 'leaky_relu':
            for i in range((args.embedding_layers-1)*2):
                self.input_acts.append(nn.LeakyReLU()) 
        if args.normalization == 'batch':
            for i in range((args.embedding_layers-1)*2):
                self.input_norm.append(nn.BatchNorm1d(hid_feats))
        elif args.normalization == 'layer':
            for i in range((args.embedding_layers-1)*2):
                self.input_norm.append(nn.LayerNorm(hid_feats))
        elif args.normalization == 'group':
            for i in range((args.embedding_layers-1)*2):
                self.input_norm.append(nn.GroupNorm(4, hid_feats))
        
        if self.opw:
            self.edges = ['entail', 'pathway']
        elif self.npw:
            self.edges = ['entail', 'occur']
        else:
            self.edges = ['entail', 'occur', 'pathway']
        
        self.conv_layers = nn.ModuleList()
        if args.residual == 'res_cat':
            self.conv_layers.append(dglnn.HeteroGraphConv(dict(zip(self.edges, [dglnn.SAGEConv(in_feats=hid_feats, out_feats=hid_feats, aggregator_type=args.agg_function, norm=None) for i in range(len(self.edges))])), aggregate='stack'))
            for i in range(args.conv_layers-1):
                self.conv_layers.append(dglnn.HeteroGraphConv(dict(zip(self.edges, [dglnn.SAGEConv(in_feats=hid_feats*2, out_feats=hid_feats, aggregator_type=args.agg_function, norm=None) for i in range(len(self.edges))])), aggregate='stack'))

        else:
            for i in range(args.conv_layers):
                self.conv_layers.append(dglnn.HeteroGraphConv(dict(zip(self.edges, [dglnn.SAGEConv(in_feats=hid_feats, out_feats=hid_feats, aggregator_type=args.agg_function, norm=None) for i in range(len(self.edges))])), aggregate='stack'))    
        
        self.conv_acts = nn.ModuleList()
        self.conv_norm = nn.ModuleList()
        if args.activation == 'gelu':
            for i in range(args.conv_layers*2):
                self.conv_acts.append(nn.GELU())
        elif args.activation == 'prelu':
            for i in range(args.conv_layers*2):
                self.conv_acts.append(nn.PReLU())
        elif args.activation == 'relu':
            for i in range(args.conv_layers*2):
                self.conv_acts.append(nn.ReLU())
        elif args.activation == 'leaky_relu':
            for i in range(args.conv_layers*2):
                self.conv_acts.append(nn.LeakyReLU())
            
        if args.normalization == 'batch':
            for i in range(args.conv_layers*len(self.edges)):
                self.conv_norm.append(nn.BatchNorm1d(hid_feats))
        elif args.normalization == 'layer':
            for i in range(args.conv_layers*len(self.edges)):
                self.conv_norm.append(nn.LayerNorm(hid_feats))
        elif args.normalization == 'group':
            for i in range(args.conv_layers*len(self.edges)):
                self.conv_norm.append(nn.GroupNorm(4, hid_feats))
        
        self.att_linears = nn.ModuleList()
        if args.pathway_aggregation == 'attention':
            for i in range(args.conv_layers):
                self.att_linears.append(nn.Linear(hid_feats, hid_feats))
        elif args.pathway_aggregation == 'one_gate':
            for i in range(args.conv_layers):
                self.att_linears.append(nn.Linear(hid_feats*3, hid_feats))
        elif args.pathway_aggregation == 'two_gate': 
            for i in range(args.conv_layers*2):
                self.att_linears.append(nn.Linear(hid_feats*2, hid_feats))
        elif args.pathway_aggregation == 'cat':
            for i in range(args.conv_layers):
                self.att_linears.append(nn.Linear(hid_feats*2, hid_feats))
                
        self.readout_linears = nn.ModuleList()
        self.readout_acts = nn.ModuleList()
        if self.nrc:
            for i in range(args.readout_layers-1):
                self.readout_linears.append(nn.Linear(hid_feats, hid_feats))
            self.readout_linears.append(nn.Linear(hid_feats, out_feats))
        else:
            for i in range(args.readout_layers-1):
                self.readout_linears.append(nn.Linear(hid_feats*args.conv_layers, hid_feats*args.conv_layers))
            self.readout_linears.append(nn.Linear(hid_feats*args.conv_layers, out_feats))
            
        if args.activation == 'gelu':
            for i in range(args.readout_layers-1):
                self.readout_acts.append(nn.GELU())
        elif args.activation == 'prelu':
            for i in range(args.readout_layers-1):
                self.readout_acts.append(nn.PReLU())
        elif args.activation == 'relu':
            for i in range(args.readout_layers-1):
                self.readout_acts.append(nn.ReLU())
        elif args.activation == 'leaky_relu':
            for i in range(args.readout_layers-1):
                self.readout_acts.append(nn.LeakyReLU())

    def attention_agg(self, layer, h0, h):
        # h: h^{l-1}, dimension: (batch, hidden)
        # feats: result from two conv(cell conv and pathway conv), stacked together; dimension: (batch, 2, hidden)
        if h.shape[1]==1:
            return self.conv_norm[layer*len(self.edges)+1](h.squeeze(1))
        elif args.pathway_aggregation == 'sum':
            return h[:, 0, :] + h[:, 1, :]
        else:
            h1 = h[:, 0, :]
            h2 = h[:, 1, :]
            
            if args.subpath_activation:
                h1 = F.leaky_relu(h1)
                h2 = F.leaky_relu(h2)
                
            h1 = self.conv_norm[layer*len(self.edges)+1](h1)
            h2 = self.conv_norm[layer*len(self.edges)+2](h2)
        
        if args.pathway_aggregation == 'attention':
            feats = torch.stack([h1, h2], 1)
            att = torch.transpose(F.softmax(torch.matmul(feats, self.att_linears[layer](h0).unsqueeze(-1)), 1), 1, 2)
            feats = torch.matmul(att, feats)
            return feats.squeeze(1)
        elif args.pathway_aggregation == 'one_gate':
            att = torch.sigmoid(self.att_linears[layer](torch.cat([h0, h1, h2], 1)))
            return att * h1 + (1-att) * h2
        elif args.pathway_aggregation == 'two_gate':
            att1 = torch.sigmoid(self.att_linears[layer*2](torch.cat([h0, h1], 1)))
            att2 = torch.sigmoid(self.att_linears[layer*2+1](torch.cat([h0, h2], 1)))
            return att1 * h1 + att2 * h2
        elif args.pathway_aggregation == 'alpha':
            return (1-args.pathway_alpha) * h1 + args.pathway_alpha * h2
        elif args.pathway_aggregation == 'cat':
            return self.att_linears[layer](torch.cat([h1, h2], 1))
        
    def conv(self, graph, layer, h, hist):
        h0 = hist[-1]
        h = self.conv_layers[layer](graph, h, mod_kwargs=dict(zip(self.edges, [{'edge_weight':  F.dropout(graph.edges[self.edges[i]].data['weight'], p=args.edge_dropout, training=self.training)} for i in range(len(self.edges))])))
        
        if args.model_dropout>0:
            h = {'feature': F.dropout(self.conv_acts[layer*2](self.attention_agg(layer, h0['feature'], h['feature'])), p=args.model_dropout, training=self.training),
             'cell': F.dropout(self.conv_acts[layer*2+1](self.conv_norm[layer*len(self.edges)](h['cell'].squeeze(1))), p=args.model_dropout, training=self.training)}
        else:
            h = {'feature': self.conv_acts[layer*2](self.attention_agg(layer, h0['feature'], h['feature'])),
             'cell': self.conv_acts[layer*2+1](self.conv_norm[layer*len(self.edges)](h['cell'].squeeze(1)))}
            
        return h
        
    def forward(self, graph):
        input1 = F.leaky_relu(self.embed_feat(graph.nodes['feature'].data['id']))
        input2 = F.leaky_relu(self.embed_cell(graph.nodes['cell'].data['id']))
        
        if not args.no_batch_features:
            input2 += F.leaky_relu(F.dropout(self.extra_encoder(batch_features), p=0.2, training=self.training))[:input2.shape[0]]
        
        if args.overlap:
            input1 += F.leaky_relu(self.ov_encoder(gex_feature))
            
        hfeat = input1
        hcell = input2
        for i in range(args.embedding_layers-1, (args.embedding_layers-1)*2):
            hfeat = self.input_linears[i](hfeat)
            hfeat = self.input_acts[i](hfeat)
            if args.normalization != 'none':
                hfeat = self.input_norm[i](hfeat)
            if args.model_dropout>0:
                hfeat = F.dropout(hfeat, p=args.model_dropout, training = self.training)
                
        for i in range(args.embedding_layers-1):
            hcell = self.input_linears[i](hcell)
            hcell = self.input_acts[i](hcell)
            if args.normalization != 'none':
                hcell = self.input_norm[i](hcell)
            if args.model_dropout>0:
                hcell = F.dropout(hcell, p=args.model_dropout, training = self.training)

        h = {'feature': hfeat, 'cell': hcell}
        hist = [h]
        
        for i in range(args.conv_layers):
            if i==0 or args.residual=='none':
                pass
            elif args.residual == 'res_add':
                if args.initial_residual:
                    h = {'feature': h['feature']+hist[0]['feature'],
                    'cell': h['cell']+hist[0]['cell']}

                else:
                    h = {'feature': h['feature']+hist[-2]['feature'],
                    'cell': h['cell']+hist[-2]['cell']}

            elif args.residual == 'res_cat':
                if args.initial_residual:
                    h = {'feature': torch.cat([h['feature'], hist[0]['feature']], 1),
                    'cell': torch.cat([h['cell'], hist[0]['cell']], 1)}
                else:
                    h = {'feature': torch.cat([h['feature'], hist[-2]['feature']], 1),
                    'cell': torch.cat([h['cell'], hist[-2]['cell']], 1)}

            h = self.conv(graph, i, h, hist)
            hist.append(h)
        
        if not self.nrc:
            h = torch.cat([i['cell'] for i in hist[1:]], 1)
        else:
            h = h['cell']
            
        for i in range(args.readout_layers-1):
            h = self.readout_linears[i](h)
            h = F.dropout(readout_acts[i](h), p=args.model_dropout, training = self.training)
        h = self.readout_linears[-1](h)
        
        if args.output_relu == 'relu':
            return F.relu(h)
        elif args.output_relu == 'leaky_relu':
            return F.leaky_relu(h)
        
        return h

device = args.device
g = g.to(device)
gtest = gtest.to(device)
train_labels = train_labels.to(device)
test_labels = test_labels.to(device)
if args.overlap:
    gex_feature = gex_feature.to(device)
if not args.no_batch_features:
    batch_features = batch_features.to(device)

#model.load_state_dict(checkpoint['model'])
#optimizer.load_state_dict(checkpoint['optimizer'])
#start_epoch = checkpoint['epoch'] + 1

model = WeightedGCN4(hid_feats=args.hidden_size, out_feats=OUTPUT_SIZE).to(device)
logger.write(str(model)+'\n')
opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
criterion = nn.MSELoss()
val = []
tr = []
minval = 100
minvep = -1

for epoch in range(args.epoch):
    logger.write(f'epoch:  {epoch}\n')
    model.train()
    logits = model(g)
    loss = criterion(logits, train_labels)
    running_loss = loss.item()
    opt.zero_grad()
    loss.backward()
    opt.step()
    torch.cuda.empty_cache()
    tr.append(math.sqrt(running_loss))
    logger.write(f'training loss:  {tr[-1]}\n')
    logger.flush()
    
    if True: #epoch % 5 == 4:
        val.append(validate(model))
        
    if epoch>4000 and val[-1]<minval:
        minval = val[-1]
        minvep = epoch
        if args.save_best:
            torch.save(model, f'{args.model_folder}/{PREFIX}.pkl')
    
    if args.early_stopping > 0 and min(val[-args.early_stopping:]) > minval:
        logger.write('Early stopped.\n')
        break

df = pd.DataFrame({'train':tr, 'val':val})
df.to_csv(f'{args.result_folder}/{PREFIX}.csv', index=False)
state = {'model':model.state_dict(), 'optimizer':opt.state_dict(), 'epoch':epoch-1}
if args.save_final:
    torch.save(state, f'{args.model_folder}/{PREFIX}.epoch{epoch}.ckpt')
logger.write(f'epoch {minvep} minimal valid:  {minval} with training:  {tr[minvep]}\n')
logger.close()