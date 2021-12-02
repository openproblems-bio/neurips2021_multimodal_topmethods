import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset,DataLoader

import anndata as ad

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
from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm

import pickle

#check gpu available
if (torch.cuda.is_available()):
    device = 'cuda:2' #switch to current device
    print('current device: gpu')
else:
    device = 'cpu'
    print('current device: cpu')


## VIASH START
dataset_path = "output/datasets/match_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_"
pretrain_path = "output/pretrain/match_modality/clue/openproblems_bmmc_cite_phase2_rna.clue_train.output_pretrain/"

par = {
    'input_train_mod1': f'{dataset_path}train_mod1.h5ad',
    'input_train_mod2': f'{dataset_path}train_mod2.h5ad',
    'output_pretrain': pretrain_path
}
meta = {
    'resources_dir': '.',
    'functionality_name': '171129'
}
## VIASH END


    




    
    

    
class ModelRegressionGex2Adt(nn.Module):
    def __init__(self, dim_mod1, dim_mod2):
        super(ModelRegressionGex2Adt, self).__init__()
        
        self.input_ = nn.Linear(dim_mod1, 512)
        self.dropout1 = nn.Dropout(p=0.20335661386636347)
        self.dropout2 = nn.Dropout(p=0.15395289261127876)
        self.dropout3 = nn.Dropout(p=0.16902655078832815)
        self.fc = nn.Linear(512, 512)
        self.fc1 = nn.Linear(512, 2048)
        self.output = nn.Linear(2048, dim_mod2)
    def forward(self, x):
       # x = self.batchswap_noise(x)
        x = F.gelu(self.input_(x))
        x = self.dropout1(x)
        x = F.gelu(self.fc(x))
        x = self.dropout2(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout3(x)
        x = F.gelu(self.output(x))
        
        return x

def train_and_valid(model, optimizer,loss_fn, dataloader_train, dataloader_test, name_model):
    best_score = 100000
    for i in range(100):
        train_losses = []
        test_losses = []
        model.train()
        for x, y in dataloader_train:
                
                optimizer.zero_grad()
                output = model(x.to(device))
                loss = torch.sqrt(loss_fn(output, y.to(device)))
                loss.backward()
                train_losses.append(loss.item())
                optimizer.step()
                
           
        model.eval()
        with torch.no_grad():
            for x, y in dataloader_test:
                output = model(x.to(device))
                output[output<0] = 0.0
                loss = torch.sqrt(loss_fn(output, y.to(device)))
                test_losses.append(loss.item())
        
        outputs = []
        targets = []
        model.eval()
        with torch.no_grad():
            for x, y in dataloader_test:
                output = model(x.to(device))
                
                outputs.append(output.detach().cpu().numpy())
                targets.append(y.detach().cpu().numpy())
        cat_outputs = np.concatenate(outputs)
        cat_targets = np.concatenate(targets)
        cat_outputs[cat_outputs<0.0] = 0
        
        if(best_score > rmse(cat_targets,cat_outputs)):
            torch.save(model.state_dict(), name_model)
            best_score = rmse(cat_targets,cat_outputs)
    print("best rmse: ", best_score)
    
def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))


print("Start train")

input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])

mod1 = input_train_mod1.var['feature_types'][0]
mod2 = input_train_mod2.var['feature_types'][0]

input_train_mod2_df = input_train_mod2.to_df()

if mod1 != "ADT":
    lsi_transformer_gex = lsiTransformer(n_components=256)
    gex_train = lsi_transformer_gex.fit_transform(input_train_mod1)
else:
    gex_train = input_train_mod1.to_df()

train_mod1, test_mod1, train_mod2, test_mod2 = train_test_split(gex_train, input_train_mod2_df, test_size=0.25, random_state=666)

if mod1 == 'ATAC' and mod2 == 'GEX':
    dataset_train = ModalityMatchingDataset(train_mod1, train_mod2)
    dataloader_train = DataLoader(dataset_train, 256, shuffle = True, num_workers = 8)

    dataset_test = ModalityMatchingDataset(test_mod1, test_mod2)
    dataloader_test = DataLoader(dataset_test, 64, shuffle = False, num_workers = 8)

    model = ModelRegressionAtac2Gex(256,13431).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00008386597445284492,weight_decay=0.000684887347727808)
        
elif mod1 == 'ADT' and mod2 == 'GEX':
    dataset_train = ModalityMatchingDataset(input_train_mod1.to_df(), input_train_mod2_df)
    dataloader_train = DataLoader(dataset_train, 64, shuffle = True, num_workers = 4)

    dataset_test = ModalityMatchingDataset(mod1_test.to_df(), mod2_test)
    dataloader_test = DataLoader(dataset_test, 32, shuffle = False, num_workers = 4)

    model = ModelRegressionAdt2Gex(134,13953).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00041, weight_decay=0.0000139)


elif mod1 == 'GEX' and mod2 == 'ADT':
    dataset_train = ModalityMatchingDataset(train_mod1, train_mod2)
    dataloader_train = DataLoader(dataset_train, 32, shuffle = True, num_workers = 8)

    dataset_test = ModalityMatchingDataset(test_mod1, test_mod2)
    dataloader_test = DataLoader(dataset_test, 64, shuffle = False, num_workers = 8)

    model = ModelRegressionGex2Adt(256,134).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.000034609210829678734,weight_decay=0.0009965881574697426)
        

elif mod1 == 'GEX' and mod2 == 'ATAC':
    dataset_train = ModalityMatchingDataset(train_mod1, train_mod2)
    dataloader_train = DataLoader(dataset_train, 64, shuffle = True, num_workers = 8)

    dataset_test = ModalityMatchingDataset(test_mod1, test_mod2)
    dataloader_test = DataLoader(dataset_test, 64, shuffle = False, num_workers = 8)

    model = ModelRegressionGex2Atac(256,10000).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001806762345275399, weight_decay=0.0004084171379280058)

loss_fn = torch.nn.MSELoss()
train_and_valid(model, optimizer, loss_fn, dataloader_train, dataloader_test, 'model.pt')

if mod1 != "ADT":
    with open('lsi_transformer.pickle', 'wb') as f:
        pickle.dump(lsi_transformer_gex, f)

print("End train")
