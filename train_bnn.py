import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data.utils as utils
sys.modules['utils'] = utils # Way to get around relative imports in utils for ZeoSynGen_dataset # https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import models.bnn as bnn
import pdb

configs = { 
            'model_type' : 'bnn',
            'split' : 'system',
            'fname': 'v3',
            'device' : 'cuda:0',
            'batch_size' : 8192,
            'n_epochs' : 10000, 
            'lr' : 1e-4,
            'model_params':{
                          'prior_mu': 0,
                          'prior_sigma': 0.3,
                        },
            }

def train_bnn(model, configs):

    # Create run folder
    assert os.path.isdir(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}") == False, 'Name already taken. Please choose another folder name.'
    os.mkdir(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}")

    # Save configs
    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/configs.json", "w") as outfile:
        json.dump(configs, outfile, indent=4)

    with open(f'data/ZeoSynGen_dataset.pkl', 'rb') as f: # load dataset
        dataset = pickle.load(f)

    train_dataset, val_dataset, _ = dataset.train_val_test_split(mode=configs['split'], both_graph_feat_present=True, random_state=0) # Note, here we filter out points with no graph/feature present for either zeolite and OSDA

    train_dataset = (train_dataset[1], train_dataset[5], train_dataset[15])
    val_dataset   = (val_dataset[1],   val_dataset[5],   val_dataset[15])

    train_loader = DataLoader(list(zip(*train_dataset)), batch_size=configs['batch_size'], shuffle=True)
    val_loader  = DataLoader(list(zip(*val_dataset)),  batch_size=configs['batch_size'], shuffle=False)

    mse_loss = nn.MSELoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl_weight = 0.01

    optimizer = optim.Adam(model.parameters(), lr=configs['lr'])

    # kl_weight = 0.1

    model.to(configs['device'])

    train_loss_list  = []
    val_loss_list    = []
    best_val_loss = 1e10
    for epoch in tqdm(range(configs['n_epochs'])):
        model.train()
        train_loss = 0.
        for (syn, zeo, osda) in train_loader:
            zeo_osda = torch.cat([zeo, osda], dim=1).to(configs['device'])
            syn = syn.to(configs['device'])
            pre = model(zeo_osda)
            mse = mse_loss(pre, syn)
            kl = kl_loss(model)
            cost = mse + kl_weight*kl
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            train_loss += cost.item()
    
        train_loss= train_loss/len(train_loader.dataset)
        train_loss_list.append(train_loss)
        print(f'Epoch: {epoch} Train Loss: {train_loss}')

        if epoch%10 == 0:
            model.eval()
            val_loss = 0.
            for (syn, zeo, osda) in val_loader:
                zeo_osda = torch.cat([zeo, osda], dim=1).to(configs['device'])
                syn = syn.to(configs['device'])
                pre = model(zeo_osda)
                mse = mse_loss(pre, syn)
                kl = kl_loss(model)
                cost = mse + kl_weight*kl

                val_loss += cost.item()

            val_loss= val_loss/len(val_loader.dataset)
            print()
            print(f'Epoch: {epoch} Val Loss: {val_loss}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print()
                print('Best model updated at Epoch {}'.format(epoch))
                torch.save(model.state_dict(), f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/best_model.pt")
                print()
            
        val_loss_list.append(val_loss)

    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/train_loss_list.pkl", 'wb') as file:
        pickle.dump(train_loss_list, file)
    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/val_loss_list.pkl", 'wb') as file:
        pickle.dump(val_loss_list, file)

if __name__ == '__main__':
    model = nn.Sequential(bnn.BayesLinear(prior_mu=configs['model_params']['prior_mu'], prior_sigma=configs['model_params']['prior_sigma'], in_features=157, out_features=128), 
                          nn.ReLU(), 
                          bnn.BayesLinear(prior_mu=configs['model_params']['prior_mu'], prior_sigma=configs['model_params']['prior_sigma'], in_features=128, out_features=12),)
    train_bnn(model, configs)
