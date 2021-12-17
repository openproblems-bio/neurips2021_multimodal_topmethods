from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy

import numpy as np
import sys

import scanpy as sc
import anndata as ad

from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import LinearRegression
from scipy.sparse import csc_matrix
import logging
from torch.utils.data.dataset import Dataset
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

## VIASH START
# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.
par = {
    'input_train_mod1': 'openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad',
    'input_train_mod2': 'openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad',
    'input_test_mod1': 'openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod1.h5ad',
    'input_test_mod2': 'openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod2.h5ad',
    'output': 'output.h5ad',
}
meta = { 'functionality_name': 'submission_171123' }
## VIASH END

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(50, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 400)
        self.bn2 = nn.BatchNorm1d(400)
        self.fc3 = nn.Linear(400, 50)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)

        x = self.relu(x)
        x = self.fc3(x)

        return x

class Net_res(nn.Module):
    def __init__(self, num_batches):
        super(Net_res, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(50, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 500)
        self.bn2 = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 500)
        self.bn3 = nn.BatchNorm1d(500)
        self.fc4 = nn.Linear(500, 500)
        self.bn4 = nn.BatchNorm1d(500)
        self.fc5 = nn.Linear(500, 50)

        self.btch_classifier = nn.Linear(500, num_batches)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(self.bn1(x))
        
        x = x + self.fc2(x)
        x = self.relu(self.bn2(x))
        
        x = x + self.fc3(x)
        x = self.relu(self.bn3(x))
        
        btch = self.btch_classifier(x)

        x = x + self.fc4(x)
        x = self.relu(self.bn4(x))
        
        x = self.fc5(x)
        
        return x, btch

class CustomDataset(Dataset):
    def __init__(self, split, X_train, X_val, X_test, y_train, y_val):
        self.split = split

        if self.split == "train":
            self.data = X_train
            self.gt = y_train
        elif self.split == "val":
            self.data = X_val
            self.gt = y_val
        else:
            self.data = X_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.split == "train":
            return self.data[idx], self.gt[idx]
        elif self.split == "val":
            return self.data[idx], 0
        else:
            return self.data[idx]

class CustomDataset_res(Dataset):
    def __init__(self, split, X_train, X_val, X_test, y_train, y_val, batches):
        self.split = split
        self.batches = batches

        if self.split == "train":
            self.data = X_train
            self.gt = y_train
        elif self.split == "val":
            self.data = X_val
            self.gt = y_val
        else:
            self.data = X_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.split == "train":
            return self.data[idx], self.gt[idx], self.batches[idx]
        elif self.split == "val":
            return self.data[idx], 0, 0
        else:
            return self.data[idx]

criterion = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()

def train_model(model, optimizer, criterion, dataloaders_dict, reverse, true_test_mod2, input_test_mod1, input_train_mod2, scheduler, num_epochs):
    best_mse = 100
    best_model = 0

    for epoch in range(num_epochs):
        y_pred = []

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, gts in tqdm(dataloaders_dict[phase]):
                inputs = inputs.cuda()
                gts = gts.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    if phase == 'train':
                        loss = criterion(outputs, gts)
                        running_loss += loss.item() * inputs.size(0)
                        loss.backward()
                        optimizer.step()
                    else:
                        y_pred.extend(outputs.cpu().numpy())


            if phase == "train":
                epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            else:
                y_pred = np.array(y_pred)
                y_pred = y_pred @ reverse

                mse = 0

                for i, sample_gt in enumerate(true_test_mod2.X):
                    mse += ((sample_gt.toarray() - y_pred[i])**2).sum()

                mse = mse / (y_pred.shape[0] * y_pred.shape[1])

                print(mse)

                if mse < best_mse:
                    best_model = copy.deepcopy(model)
                    best_mse = mse
    print("Best MSE: ", best_mse)
    
    return best_model

def train_model_res(model, optimizer, criterion, criterion2, dataloaders_dict, reverse, true_test_mod2, input_test_mod1, input_train_mod2, scheduler, num_epochs):
    best_mse = 100
    best_model = 0

    for epoch in range(num_epochs):
        y_pred = []


        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, gts, btch in tqdm(dataloaders_dict[phase]):
                inputs = inputs.cuda()
                gts = gts.cuda()
                btch = btch.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs, out_btch = model(inputs)

                    if phase == 'train':
                        loss1 = criterion(outputs, gts)
                        running_loss += loss1.item() * inputs.size(0)
                        loss2 = criterion2(out_btch, btch)

                        loss = 1 / 9 * loss1 + 8 / 9 * loss2
                        loss.backward()
                        optimizer.step()
                    else:
                        y_pred.extend(outputs.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == "val":
                y_pred = np.array(y_pred)
                y_pred = y_pred @ reverse

                mse = 0

                for i, sample_gt in enumerate(true_test_mod2.X):
                    mse += ((sample_gt.toarray() - y_pred[i])**2).sum()

                mse = mse / (y_pred.shape[0] * y_pred.shape[1])

                print(mse**0.5)

                if mse < best_mse:
                    best_model = copy.deepcopy(model)
                    best_mse = mse
    print("Best RMSE: ", best_mse**0.5)
    
    return best_model

def infer_res(model, dataloader, input_test_mod1, input_train_mod1, input_train_mod2, reverse):
    y_pred = []
    model.eval()

    for inputs in tqdm(dataloader):
        inputs = inputs.cuda()

        with torch.set_grad_enabled(False):
            outputs, _ = model(inputs)

            y_pred.extend(outputs.cpu().numpy())

    y_pred = np.array(y_pred)
    y_pred = y_pred @ reverse

    return y_pred
    
def infer(model, dataloader, input_test_mod1, input_train_mod1, input_train_mod2, reverse):
    y_pred = []
    model.eval()

    for inputs in tqdm(dataloader):
        inputs = inputs.cuda()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            y_pred.extend(outputs.cpu().numpy())

    y_pred = np.array(y_pred)
    y_pred = y_pred @ reverse

    return y_pred

logging.info('Reading `h5ad` files...')
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])

dataset_id = input_train_mod1.uns['dataset_id']

input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
final_input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

batches = set(input_train_mod1.obs["batch"])
batch_dict = {batch:i for i, batch in enumerate(batches)}
y = []

for i in range(input_train_mod1.n_obs):
    y.append(int(batch_dict[input_train_mod1.obs["batch"][i]]))

fold = 0

X = input_train_mod1.obs
batches = np.array(y)

inp_train_mod1 = input_train_mod1.copy()
inp_train_mod2 = input_train_mod2.copy()

# TODO: implement own method
out1, out2 = 0, 0

params1 = {'learning_rate': 0.3, 
          'depth': 6, 
          'l2_leaf_reg': 3, 
          'loss_function': 'MultiRMSE', 
          'eval_metric': 'MultiRMSE', 
          'task_type': 'CPU', 
          'iterations': 150,
          'od_type': 'Iter', 
          'boosting_type': 'Plain', 
          'bootstrap_type': 'Bernoulli', 
          'allow_const_label': True, 
          'random_state': 1
         }

params2 = {'learning_rate': 0.2, 
          'depth': 7, 
          'l2_leaf_reg': 4, 
          'loss_function': 'MultiRMSE', 
          'eval_metric': 'MultiRMSE', 
          'task_type': 'CPU', 
          'iterations': 200,
          'od_type': 'Iter', 
          'boosting_type': 'Plain', 
          'bootstrap_type': 'Bayesian', 
          'allow_const_label': True, 
          'random_state': 1
         }

if "adt2gex" in dataset_id:
    for train_index, test_index in skf.split(X, batches):
        print(fold)
        fold += 1

        input_test_mod1 = inp_train_mod1[test_index, :]
        true_test_mod2 = inp_train_mod2[test_index, :]

        input_train_mod1 = inp_train_mod1[train_index, :]
        input_train_mod2 = inp_train_mod2[train_index, :]
    
        input_mod1 = ad.concat(
                    {"train": input_train_mod1, "val": input_test_mod1, "test": final_input_test_mod1},
                    axis=0,
                    join="outer",
                    label="group",
                    fill_value=0,
                    index_unique="-",
                )

        # Do PCA on the input data
        logging.info('Performing dimensionality reduction on modality 1 values...')
        embedder_mod1 = TruncatedSVD(n_components=50)
        mod1_pca = embedder_mod1.fit_transform(input_mod1.X)

        logging.info('Performing dimensionality reduction on modality 2 values...')
        embedder_mod2 = TruncatedSVD(n_components=50)
        mod2_pca = embedder_mod2.fit_transform(input_train_mod2.X)

        # split dimred back up
        X_train = mod1_pca[input_mod1.obs['group'] == 'train']
        X_test = mod1_pca[input_mod1.obs['group'] == 'test']
        y_train = mod2_pca

        # Get all responses of the training data set to fit the
        # KNN regressor later on.
        # Make sure to use `toarray()` because the output might
        # be sparse and `KNeighborsRegressor` cannot handle it.

        reg = CatBoostRegressor(**params1)

        # Train the model on the PCA reduced modality 1 and 2 data
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        # Project the predictions back to the modality 2 feature space
        y_pred = y_pred @ embedder_mod2.components_
        out1 += y_pred
    
        reg = CatBoostRegressor(**params2)

        # Train the model on the PCA reduced modality 1 and 2 data
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        # Project the predictions back to the modality 2 feature space
        y_pred = y_pred @ embedder_mod2.components_
        out2 += y_pred
    
    # Store as sparse matrix to be efficient. Note that this might require
    # different classifiers/embedders before-hand. Not every class is able
    # to support such data structures.
    y_pred_cat = (out1 / 10 + out2 / 10) / 2

    out_net = 0
    out_net_res = 0

    for train_index, test_index in skf.split(X, batches):
        print(fold)
        #model = Net()
        model = Net().cuda()
        lr = 1e-2
        optimizer_ft = optim.Adam(model.parameters(), lr=lr)
    
        fold += 1

        input_test_mod1 = inp_train_mod1[test_index, :]
        true_test_mod2 = inp_train_mod2[test_index, :]

        input_train_mod1 = inp_train_mod1[train_index, :]
        input_train_mod2 = inp_train_mod2[train_index, :]

        input_mod1 = ad.concat(
                {"train": input_train_mod1, "val": input_test_mod1, "test": final_input_test_mod1},
                axis=0,
                join="outer",
                label="group",
                fill_value=0,
                index_unique="-",
            )

        # Do PCA on the input data
        print('Performing dimensionality reduction on modality 1 values...')
        embedder_mod1 = TruncatedSVD(n_components=50)

        print(input_mod1.X.shape)
        print(input_train_mod2.X.shape)

        mod1_pca = embedder_mod1.fit_transform(input_mod1.X)

        print('Performing dimensionality reduction on modality 2 values...')
        embedder_mod2 = TruncatedSVD(n_components=50)
        mod2_pca = embedder_mod2.fit_transform(input_train_mod2.X)

        # split dimred mod 1 back up for training
        X_train = mod1_pca[input_mod1.obs['group'] == 'train']
        X_val = mod1_pca[input_mod1.obs['group'] == 'val']
        X_test = mod1_pca[input_mod1.obs['group'] == 'test']
        y_train = mod2_pca
        y_val = true_test_mod2

        data = {x: CustomDataset(x, X_train, X_val, X_test, y_train, y_val) for x in ['train', 'val', 'test']}

        dataloaders_dict = {"train": torch.utils.data.DataLoader(data["train"], batch_size=100, shuffle=True, num_workers=8),
                        "val": torch.utils.data.DataLoader(data["val"], batch_size=100, shuffle=False, num_workers=8),
                        "test": torch.utils.data.DataLoader(data["test"], batch_size=100, shuffle=False, num_workers=8)}

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, patience=6, factor=0.1, verbose=True)
        best_model_net = train_model(model, optimizer_ft, criterion, dataloaders_dict, embedder_mod2.components_, y_val, input_test_mod1, input_train_mod2, scheduler, num_epochs=25)

        y_pred = infer(best_model_net, dataloaders_dict["test"], final_input_test_mod1, inp_train_mod1, inp_train_mod2, embedder_mod2.components_)
        out_net += y_pred
        train_btchs = batches[train_index]
    
        data = {x: CustomDataset_res(x, X_train, X_val, X_test, y_train, y_val, train_btchs) for x in ['train', 'val', 'test']}

        dataloaders_dict = {"train":
                            torch.utils.data.DataLoader(data["train"], batch_size=100, shuffle=True, num_workers=8),
                        "val": torch.utils.data.DataLoader(data["val"], batch_size=100, shuffle=False, num_workers=8),
                        "test": torch.utils.data.DataLoader(data["test"], batch_size=100, shuffle=False, num_workers=8)
                        }
    
        model = Net_res(num_batches=len(set(train_btchs))).cuda()
        #model = Net_res(num_batches=len(set(train_btchs)))
        lr = 1e-2
        optimizer_ft = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, patience=6, factor=0.1, verbose=True)
        best_model_net_res = train_model_res(model, optimizer_ft, criterion, criterion2, dataloaders_dict, embedder_mod2.components_, y_val, input_test_mod1, input_train_mod2, scheduler, num_epochs=25)

        y_pred = infer_res(best_model_net_res, dataloaders_dict["test"], final_input_test_mod1, inp_train_mod1, inp_train_mod2, embedder_mod2.components_)
        out_net_res += y_pred
    

    y_pred_mlp = (out_net / 10 + out_net_res / 10) / 2
    
    out_rf = 0

    for train_index, test_index in skf.split(X, y):
        print(fold)
        fold += 1

        input_test_mod1 = inp_train_mod1[test_index, :]
        true_test_mod2 = inp_train_mod2[test_index, :]

        input_train_mod1 = inp_train_mod1[train_index, :]
        input_train_mod2 = inp_train_mod2[train_index, :]
    
        input_mod1 = ad.concat(
                {"train": input_train_mod1, "val": input_test_mod1, "test": final_input_test_mod1},
                axis=0,
                join="outer",
                label="group",
                fill_value=0,
                index_unique="-",
            )

        # Do PCA on the input data
        logging.info('Performing dimensionality reduction on modality 1 values...')
        embedder_mod1 = TruncatedSVD(n_components=50)
        mod1_pca = embedder_mod1.fit_transform(input_mod1.X)

        logging.info('Performing dimensionality reduction on modality 2 values...')
        embedder_mod2 = TruncatedSVD(n_components=50)
        mod2_pca = embedder_mod2.fit_transform(input_train_mod2.X)

        # split dimred back up
        X_train = mod1_pca[input_mod1.obs['group'] == 'train']
        X_test = mod1_pca[input_mod1.obs['group'] == 'test']
        y_train = mod2_pca

        # Get all responses of the training data set to fit the
        # KNN regressor later on.
        # Make sure to use `toarray()` because the output might
        # be sparse and `KNeighborsRegressor` cannot handle it.

        logging.info('Running Linear regression...')
    
        reg = RandomForestRegressor()

        # Train the model on the PCA reduced modality 1 and 2 data
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        # Project the predictions back to the modality 2 feature space
        y_pred = y_pred @ embedder_mod2.components_
    
        out_rf += y_pred

    y_pred_rf = out_rf / 10
    
    y_pred = (y_pred_cat + y_pred_mlp + y_pred_rf) / 3

    y_pred = csc_matrix(y_pred)

    adata = ad.AnnData(
        X=y_pred,
        obs=final_input_test_mod1.obs,
        var=inp_train_mod2.var,
        uns={
            'dataset_id': dataset_id,
            'method_id': meta["functionality_name"],
        },
    )
    
    logging.info('Storing annotated data...')
    adata.write_h5ad(par['output'], compression = "gzip")
elif "gex2adt" in dataset_id:
    for train_index, test_index in skf.split(X, batches):
        print(fold)
        fold += 1

        input_test_mod1 = inp_train_mod1[test_index, :]
        true_test_mod2 = inp_train_mod2[test_index, :]

        input_train_mod1 = inp_train_mod1[train_index, :]
        input_train_mod2 = inp_train_mod2[train_index, :]
    
        input_mod1 = ad.concat(
                    {"train": input_train_mod1, "val": input_test_mod1, "test": final_input_test_mod1},
                    axis=0,
                    join="outer",
                    label="group",
                    fill_value=0,
                    index_unique="-",
                )

        # Do PCA on the input data
        logging.info('Performing dimensionality reduction on modality 1 values...')
        embedder_mod1 = TruncatedSVD(n_components=50)
        mod1_pca = embedder_mod1.fit_transform(input_mod1.X)

        logging.info('Performing dimensionality reduction on modality 2 values...')
        embedder_mod2 = TruncatedSVD(n_components=50)
        mod2_pca = embedder_mod2.fit_transform(input_train_mod2.X)

        # split dimred back up
        X_train = mod1_pca[input_mod1.obs['group'] == 'train']
        X_test = mod1_pca[input_mod1.obs['group'] == 'test']
        y_train = mod2_pca

        # Get all responses of the training data set to fit the
        # KNN regressor later on.
        # Make sure to use `toarray()` because the output might
        # be sparse and `KNeighborsRegressor` cannot handle it.

        reg = CatBoostRegressor(**params1)

        # Train the model on the PCA reduced modality 1 and 2 data
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        # Project the predictions back to the modality 2 feature space
        y_pred = y_pred @ embedder_mod2.components_
        out1 += y_pred
    
        reg = CatBoostRegressor(**params2)

        # Train the model on the PCA reduced modality 1 and 2 data
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        # Project the predictions back to the modality 2 feature space
        y_pred = y_pred @ embedder_mod2.components_
        out2 += y_pred
    
    # Store as sparse matrix to be efficient. Note that this might require
    # different classifiers/embedders before-hand. Not every class is able
    # to support such data structures.
    y_pred_cat = (out1 / 10 + out2 / 10) / 2

    out_net = 0
    out_net_res = 0

    for train_index, test_index in skf.split(X, batches):
        print(fold)
        #model = Net()
        model = Net().cuda()
        lr = 1e-2
        optimizer_ft = optim.Adam(model.parameters(), lr=lr)
    
        fold += 1

        input_test_mod1 = inp_train_mod1[test_index, :]
        true_test_mod2 = inp_train_mod2[test_index, :]

        input_train_mod1 = inp_train_mod1[train_index, :]
        input_train_mod2 = inp_train_mod2[train_index, :]

        input_mod1 = ad.concat(
                {"train": input_train_mod1, "val": input_test_mod1, "test": final_input_test_mod1},
                axis=0,
                join="outer",
                label="group",
                fill_value=0,
                index_unique="-",
            )

        # Do PCA on the input data
        print('Performing dimensionality reduction on modality 1 values...')
        embedder_mod1 = TruncatedSVD(n_components=50)

        print(input_mod1.X.shape)
        print(input_train_mod2.X.shape)

        mod1_pca = embedder_mod1.fit_transform(input_mod1.X)

        print('Performing dimensionality reduction on modality 2 values...')
        embedder_mod2 = TruncatedSVD(n_components=50)
        mod2_pca = embedder_mod2.fit_transform(input_train_mod2.X)

        # split dimred mod 1 back up for training
        X_train = mod1_pca[input_mod1.obs['group'] == 'train']
        X_val = mod1_pca[input_mod1.obs['group'] == 'val']
        X_test = mod1_pca[input_mod1.obs['group'] == 'test']
        y_train = mod2_pca
        y_val = true_test_mod2

        data = {x: CustomDataset(x, X_train, X_val, X_test, y_train, y_val) for x in ['train', 'val', 'test']}

        dataloaders_dict = {"train": torch.utils.data.DataLoader(data["train"], batch_size=100, shuffle=True, num_workers=8),
                        "val": torch.utils.data.DataLoader(data["val"], batch_size=100, shuffle=False, num_workers=8),
                        "test": torch.utils.data.DataLoader(data["test"], batch_size=100, shuffle=False, num_workers=8)}

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, patience=6, factor=0.1, verbose=True)
        best_model_net = train_model(model, optimizer_ft, criterion, dataloaders_dict, embedder_mod2.components_, y_val, input_test_mod1, input_train_mod2, scheduler, num_epochs=25)

        y_pred = infer(best_model_net, dataloaders_dict["test"], final_input_test_mod1, inp_train_mod1, inp_train_mod2, embedder_mod2.components_)
        out_net += y_pred
        train_btchs = batches[train_index]
    
        data = {x: CustomDataset_res(x, X_train, X_val, X_test, y_train, y_val, train_btchs) for x in ['train', 'val', 'test']}

        dataloaders_dict = {"train":
                            torch.utils.data.DataLoader(data["train"], batch_size=100, shuffle=True, num_workers=8),
                        "val": torch.utils.data.DataLoader(data["val"], batch_size=100, shuffle=False, num_workers=8),
                        "test": torch.utils.data.DataLoader(data["test"], batch_size=100, shuffle=False, num_workers=8)
                        }
    
        model = Net_res(num_batches=len(set(train_btchs))).cuda()
        #model = Net_res(num_batches=len(set(train_btchs)))
        lr = 1e-2
        optimizer_ft = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, patience=6, factor=0.1, verbose=True)
        best_model_net_res = train_model_res(model, optimizer_ft, criterion, criterion2, dataloaders_dict, embedder_mod2.components_, y_val, input_test_mod1, input_train_mod2, scheduler, num_epochs=25)

        y_pred = infer_res(best_model_net_res, dataloaders_dict["test"], final_input_test_mod1, inp_train_mod1, inp_train_mod2, embedder_mod2.components_)
        out_net_res += y_pred

    y_pred_mlp = (out_net / 10 + out_net_res / 10) / 2
    y_pred = (y_pred_cat + y_pred_mlp) / 2

    y_pred = csc_matrix(y_pred)

    adata = ad.AnnData(
        X=y_pred,
        obs=final_input_test_mod1.obs,
        var=inp_train_mod2.var,
        uns={
            'dataset_id': dataset_id,
            'method_id': meta["functionality_name"],
        },
    )
    
    logging.info('Storing annotated data...')
    adata.write_h5ad(par['output'], compression = "gzip")
elif "atac2gex" in dataset_id:
    out_knn = 0

    for train_index, test_index in skf.split(X, y):
        print(fold)
        fold += 1

        input_test_mod1 = inp_train_mod1[test_index, :]
        true_test_mod2 = inp_train_mod2[test_index, :]

        input_train_mod1 = inp_train_mod1[train_index, :]
        input_train_mod2 = inp_train_mod2[train_index, :]
    
        input_mod1 = ad.concat(
                {"train": input_train_mod1, "val": input_test_mod1, "test": final_input_test_mod1},
                axis=0,
                join="outer",
                label="group",
                fill_value=0,
                index_unique="-",
            )

        # Do PCA on the input data
        logging.info('Performing dimensionality reduction on modality 1 values...')
        embedder_mod1 = TruncatedSVD(n_components=50)
        mod1_pca = embedder_mod1.fit_transform(input_mod1.X)

        logging.info('Performing dimensionality reduction on modality 2 values...')
        embedder_mod2 = TruncatedSVD(n_components=50)
        mod2_pca = embedder_mod2.fit_transform(input_train_mod2.X)

        # split dimred back up
        X_train = mod1_pca[input_mod1.obs['group'] == 'train']
        X_test = mod1_pca[input_mod1.obs['group'] == 'test']
        y_train = mod2_pca

        # Get all responses of the training data set to fit the
        # KNN regressor later on.
        # Make sure to use `toarray()` because the output might
        # be sparse and `KNeighborsRegressor` cannot handle it.

        logging.info('Running Linear regression...')
    
        reg = KNeighborsRegressor(n_neighbors=25, metric='minkowski')

        # Train the model on the PCA reduced modality 1 and 2 data
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        # Project the predictions back to the modality 2 feature space
        y_pred = y_pred @ embedder_mod2.components_
    
        out_knn += y_pred

    y_pred_knn = out_knn / 10
    
    out_rf = 0

    for train_index, test_index in skf.split(X, y):
        print(fold)
        fold += 1

        input_test_mod1 = inp_train_mod1[test_index, :]
        true_test_mod2 = inp_train_mod2[test_index, :]

        input_train_mod1 = inp_train_mod1[train_index, :]
        input_train_mod2 = inp_train_mod2[train_index, :]
    
        input_mod1 = ad.concat(
                {"train": input_train_mod1, "val": input_test_mod1, "test": final_input_test_mod1},
                axis=0,
                join="outer",
                label="group",
                fill_value=0,
                index_unique="-",
            )

        # Do PCA on the input data
        logging.info('Performing dimensionality reduction on modality 1 values...')
        embedder_mod1 = TruncatedSVD(n_components=50)
        mod1_pca = embedder_mod1.fit_transform(input_mod1.X)

        logging.info('Performing dimensionality reduction on modality 2 values...')
        embedder_mod2 = TruncatedSVD(n_components=50)
        mod2_pca = embedder_mod2.fit_transform(input_train_mod2.X)

        # split dimred back up
        X_train = mod1_pca[input_mod1.obs['group'] == 'train']
        X_test = mod1_pca[input_mod1.obs['group'] == 'test']
        y_train = mod2_pca

        # Get all responses of the training data set to fit the
        # KNN regressor later on.
        # Make sure to use `toarray()` because the output might
        # be sparse and `KNeighborsRegressor` cannot handle it.

        logging.info('Running Linear regression...')
    
        reg = RandomForestRegressor()

        # Train the model on the PCA reduced modality 1 and 2 data
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        # Project the predictions back to the modality 2 feature space
        y_pred = y_pred @ embedder_mod2.components_
    
        out_rf += y_pred

    y_pred_rf = out_rf / 10
    
    y_pred = 0.45 * y_pred_rf + 0.55 * y_pred_knn
    y_pred = csc_matrix(y_pred)

    adata = ad.AnnData(
        X=y_pred,
       obs=final_input_test_mod1.obs,
       var=inp_train_mod2.var,
       uns={
           'dataset_id': dataset_id,
           'method_id': meta["functionality_name"],
       },
    )
    
    logging.info('Storing annotated data...')
    adata.write_h5ad(par['output'], compression = "gzip")
else:
    out_knn = 0

    for train_index, test_index in skf.split(X, y):
        print(fold)
        fold += 1

        input_test_mod1 = inp_train_mod1[test_index, :]
        true_test_mod2 = inp_train_mod2[test_index, :]

        input_train_mod1 = inp_train_mod1[train_index, :]
        input_train_mod2 = inp_train_mod2[train_index, :]
    
        input_mod1 = ad.concat(
                {"train": input_train_mod1, "val": input_test_mod1, "test": final_input_test_mod1},
                axis=0,
                join="outer",
                label="group",
                fill_value=0,
                index_unique="-",
            )

        # Do PCA on the input data
        logging.info('Performing dimensionality reduction on modality 1 values...')
        embedder_mod1 = TruncatedSVD(n_components=50)
        mod1_pca = embedder_mod1.fit_transform(input_mod1.X)

        logging.info('Performing dimensionality reduction on modality 2 values...')
        embedder_mod2 = TruncatedSVD(n_components=50)
        mod2_pca = embedder_mod2.fit_transform(input_train_mod2.X)

        # split dimred back up
        X_train = mod1_pca[input_mod1.obs['group'] == 'train']
        X_test = mod1_pca[input_mod1.obs['group'] == 'test']
        y_train = mod2_pca

        # Get all responses of the training data set to fit the
        # KNN regressor later on.
        # Make sure to use `toarray()` because the output might
        # be sparse and `KNeighborsRegressor` cannot handle it.

        logging.info('Running Linear regression...')
    
        reg = KNeighborsRegressor(n_neighbors=25, metric='minkowski')

        # Train the model on the PCA reduced modality 1 and 2 data
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        # Project the predictions back to the modality 2 feature space
        y_pred = y_pred @ embedder_mod2.components_
    
        out_knn += y_pred

    y_pred = out_knn / 10
    y_pred = csc_matrix(y_pred)

    adata = ad.AnnData(
        X=y_pred,
       obs=final_input_test_mod1.obs,
       var=inp_train_mod2.var,
       uns={
           'dataset_id': dataset_id,
           'method_id': meta["functionality_name"],
       },
    )
    
    logging.info('Storing annotated data...')
    adata.write_h5ad(par['output'], compression = "gzip")
