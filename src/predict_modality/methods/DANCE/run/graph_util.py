import os
import numpy as np
import pandas as pd
import logging
import math
import json
import re
import argparse
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import anndata as ad
import matplotlib.pyplot as plt

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
import pickle
from collections import defaultdict

def graph_construction(meta, train_mod1, train_mod2, test_mod1, pretrain_path, bidirect = True):
    # Pathway Features
    mod1 = train_mod1.var['feature_types'][0]
    mod2 = train_mod2.var['feature_types'][0]
    input_train_mod1 = train_mod1.X
    input_train_mod2 = train_mod2.X
    input_test_mod1 = test_mod1.X
    
    TRAIN_SIZE = train_mod1.shape[0]
    TEST_SIZE = test_mod1.shape[0]
    FEATURE_SIZE = train_mod1.shape[1]
    CELL_SIZE = TRAIN_SIZE + TEST_SIZE
    
    uu=[]
    vv=[]
    ee=[]

    if mod1 == 'GEX' and mod2 == 'ADT':
        uu, vv, ee = pickle.load(open(os.path.join(pretrain_path, 'pw.pkl'), 'rb'))
        ee = [e.item() for e in ee]
    elif mod1 == 'GEX' and mod2 == 'ATAC':
        uu, vv, ee = pickle.load(open(os.path.join(pretrain_path, 'pw_multiome.pkl'), 'rb'))
        ee = [e.item() for e in ee]
    else:
        with open(os.path.join(pretrain_path, 'h.all.v7.4.entrez.gmt')) as gmt:
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

        with open(os.path.join(pretrain_path, 'h.all.v7.4.symbols.gmt')) as gmt:
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

    # Batch Features

    cells = []
    columns = ['cell_mean', 'cell_std', 'nonzero_25%', 'nonzero_50%', 'nonzero_75%', 'nonzero_max', 'nonzero_count', 'nonzero_mean', 'nonzero_std', 'batch']

    bcl = list(train_mod1.obs['batch'])
    for i, cell in enumerate(input_train_mod1):
        cell = cell.toarray()
        nz = cell[np.nonzero(cell)]
        cells.append([cell.mean(), cell.std(), np.percentile(nz, 25), np.percentile(nz, 50), np.percentile(nz, 75), cell.max(), len(nz)/1000, nz.mean(), nz.std(), bcl[i]])

    bcl = list(test_mod1.obs['batch'])
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

    for b in train_mod1.obs['batch']:
        batch_features.append(batch_source[b2i[b]])

    for b in test_mod1.obs['batch']:
        batch_features.append(batch_source[b2i[b]])

    batch_features = torch.tensor(batch_features).float()

    # Graph construction
    u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0]+i) for i, t in enumerate(input_train_mod1)] +\
                       [np.array(t.nonzero()[0]+i+TRAIN_SIZE) for i, t in enumerate(input_test_mod1)], axis=0))
    v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1] +\
                       [np.array(t.nonzero()[1]) for t in input_test_mod1], axis=0))

    if bidirect:
        graph_data = {
            ('cell', 'occur', 'feature'): (u, v), #(torch.from_numpy(np.concatenate([np.array(t.nonzero()[0]+i) for i, t in enumerate(input_train_mod1)], axis=0)), torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1], axis=0))), 
            ('feature', 'entail', 'cell'): (v, u),
            ('feature', 'pathway', 'feature'): (uu, vv),
        }
    else:
        graph_data = {
            ('cell', 'occur', 'feature'): (torch.from_numpy(np.concatenate([np.array(t.nonzero()[0]+i) for i, t in enumerate(input_train_mod1)], axis=0)), torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1], axis=0))), 
            ('feature', 'entail', 'cell'): (v, u),
            ('feature', 'pathway', 'feature'): (uu, vv),
        }
    g = dgl.heterograph(graph_data)

    g.nodes['cell'].data['id'] = torch.ones(CELL_SIZE).long()
    g.nodes['feature'].data['id'] = torch.arange(FEATURE_SIZE).long()
    g.edges['entail'].data['weight'] = torch.from_numpy(np.concatenate([input_train_mod1.tocsr().data, input_test_mod1.tocsr().data], axis=0)).float()
    if bidirect:
        g.edges['occur'].data['weight'] = torch.from_numpy(np.concatenate([input_train_mod1.tocsr().data, input_test_mod1.tocsr().data], axis=0)).float() #torch.from_numpy(input_train_mod1.tocsr().data).float() #
    else:
        torch.from_numpy(input_train_mod1.tocsr().data).float()
    g.edges['pathway'].data['weight'] = torch.tensor(ee).float()
    
    return g, batch_features
    
class WeightedGCN4(nn.Module):
    def __init__(self, FEATURE_SIZE, hid_feats, out_feats, bf_size, args):
        super().__init__()
        
        self.extra_encoder = nn.Linear(bf_size, hid_feats)
        self.args = args

        self.embed_cell = nn.Embedding(2, hid_feats)
        self.embed_feat = nn.Embedding(FEATURE_SIZE, hid_feats)

        self.input_linears = nn.ModuleList()
        self.input_acts = nn.ModuleList()
        self.input_norm = nn.ModuleList()
        for i in range(2):
            self.input_linears.append(nn.Linear(hid_feats, hid_feats))
        if self.args['activation'] == 'gelu':
            for i in range(2):
                self.input_acts.append(nn.GELU())
        elif self.args['activation'] == 'leaky_relu':
            for i in range(2):
                self.input_acts.append(nn.LeakyReLU()) 

        for i in range(2):
            self.input_norm.append(nn.GroupNorm(4, hid_feats))

        self.edges = ['entail', 'occur', 'pathway']

        self.conv_layers = nn.ModuleList()

        self.conv_layers.append(dglnn.HeteroGraphConv(dict(zip(self.edges, [dglnn.SAGEConv(in_feats=hid_feats, out_feats=hid_feats, aggregator_type=self.args['agg_function'], norm=None) for i in range(len(self.edges))])), aggregate='stack'))
        for i in range(self.args['conv_layers']-1):
            self.conv_layers.append(dglnn.HeteroGraphConv(dict(zip(self.edges, [dglnn.SAGEConv(in_feats=hid_feats*2, out_feats=hid_feats, aggregator_type=self.args['agg_function'], norm=None) for i in range(len(self.edges))])), aggregate='stack'))

        self.conv_acts = nn.ModuleList()
        self.conv_norm = nn.ModuleList()
        if self.args['activation'] == 'gelu':
            for i in range(self.args['conv_layers']*2):
                self.conv_acts.append(nn.GELU())
        elif self.args['activation'] == 'leaky_relu':
            for i in range(self.args['conv_layers']*2):
                self.conv_acts.append(nn.LeakyReLU())

        for i in range(self.args['conv_layers']*len(self.edges)):
            self.conv_norm.append(nn.GroupNorm(4, hid_feats))

        self.att_linears = nn.ModuleList()
        self.readout_linears = nn.ModuleList()
        self.readout_acts = nn.ModuleList()
        self.readout_linears.append(nn.Linear(hid_feats*self.args['conv_layers'], out_feats))

        if self.args['activation'] == 'gelu':
            for i in range(self.args['readout_layers']-1):
                self.readout_acts.append(nn.GELU())
        elif self.args['activation'] == 'leaky_relu':
            for i in range(self.args['readout_layers']-1):
                self.readout_acts.append(nn.LeakyReLU())

    def attention_agg(self, layer, h0, h):
        # h: h^{l-1}, dimension: (batch, hidden)
        # feats: result from two conv(cell conv and pathway conv), stacked together; dimension: (batch, 2, hidden)
        if h.shape[1]==1:
            return self.conv_norm[layer*len(self.edges)+1](h.squeeze(1))
        else:
            h1 = self.conv_norm[layer*len(self.edges)+1](h[:, 0, :])
            h2 = self.conv_norm[layer*len(self.edges)+2](h[:, 1, :])

        return (1-self.args['pathway_alpha']) * h1 + self.args['pathway_alpha'] * h2

    def conv(self, graph, layer, h, hist):
        h0 = hist[-1]
        h = self.conv_layers[layer](graph, h, mod_kwargs=dict(zip(self.edges, [{'edge_weight':  F.dropout(graph.edges[self.edges[i]].data['weight'], p=self.args['edge_dropout'], training=self.training)} for i in range(len(self.edges))])))

        h = {'feature': F.dropout(self.conv_acts[layer*2](self.attention_agg(layer, h0['feature'], h['feature'])), p=self.args['model_dropout'], training=self.training),
         'cell': F.dropout(self.conv_acts[layer*2+1](self.conv_norm[layer*len(self.edges)](h['cell'].squeeze(1))), p=self.args['model_dropout'], training=self.training)}
        return h

    def forward(self, graph, batch_features):
        input1 = F.leaky_relu(self.embed_feat(graph.nodes['feature'].data['id']))
        input2 = F.leaky_relu(self.embed_cell(graph.nodes['cell'].data['id']))
        input2 += F.leaky_relu(F.dropout(self.extra_encoder(batch_features), p=0.2, training=self.training))[:input2.shape[0]]

        hfeat = input1
        hcell = input2
        for i in range(1, 2):
            hfeat = self.input_linears[i](hfeat)
            hfeat = self.input_acts[i](hfeat)
            hfeat = self.input_norm[i](hfeat)
            hfeat = F.dropout(hfeat, p=self.args['model_dropout'], training = self.training)

        for i in range(1):
            hcell = self.input_linears[i](hcell)
            hcell = self.input_acts[i](hcell)
            hcell = self.input_norm[i](hcell)
            hcell = F.dropout(hcell, p=self.args['model_dropout'], training = self.training)

        h = {'feature': hfeat, 'cell': hcell}
        hist = [h]

        for i in range(self.args['conv_layers']):
            if i>0:
                h = {'feature': torch.cat([h['feature'], hist[0]['feature']], 1),
                    'cell': torch.cat([h['cell'], hist[0]['cell']], 1)}

            h = self.conv(graph, i, h, hist)
            hist.append(h)

        h = torch.cat([i['cell'] for i in hist[1:]], 1)
        h = self.readout_linears[-1](h)

        return h