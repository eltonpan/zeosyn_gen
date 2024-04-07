import os
import sys
import pdb
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data.utils as utils
sys.modules['utils'] = utils # Way to get around relative imports in utils for ZeoSynGen_dataset # https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from models.nf import ConditionalNormalizingFlow
from torch.distributions import Normal

configs = { 
            'model_type' : 'nf',
            'split' : 'system',
            'fname': 'v0',
            'device' : 'cuda:0',
            'batch_size' : 2048,
            'n_epochs' : 3000, # 2e4
            'lr' : 1e-4,
            'model_params':{
                        'num_flows': 32,
                        'hidden_size': 1024,
                        'zeo_h_dims': 64, 
                        'osda_h_dims': 64, 
                        'syn_dims': 12, 
                        'zeo_feat_dims': 143, 
                        'osda_feat_dims': 14,
                        },
            }

def train_nf(model, configs):

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

    model.to(configs['device'])

    best_model = None # Model to be outputted based on val loss
    train_loss_list  = []
    val_loss_list    = []

    best_val_loss = 1e10
    optimizer = optim.Adam(model.parameters(), lr=configs['lr'])

    for epoch in tqdm(range(configs['n_epochs'])): 

        # Training
        model.train()
        train_loss        = 0.
        recons_loss_total = 0.
        kld_loss_total    = 0.
        for (syn, zeo, osda) in train_loader:
            syn, zeo, osda = syn.to(configs['device']), zeo.to(configs['device']), osda.to(configs['device'])

            # Forward pass
            z, log_det = model(syn, zeo, osda)
            log_prob = Normal(0, 1).log_prob(z).sum(dim=1) + log_det

            # Compute the loss (negative log-likelihood)
            loss = -log_prob.mean()

            train_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch: {} Train Loss: {:.5f}'.format(
        epoch, 
        train_loss/len(train_loader.dataset), ))

        train_loss_list.append(train_loss/len(train_loader.dataset))

        # Validation
        if epoch%5 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.
                for (syn, zeo, osda) in val_loader:
                    syn, zeo, osda = syn.to(configs['device']), zeo.to(configs['device']), osda.to(configs['device'])

                    # Forward pass
                    z, log_det = model(syn, zeo, osda)
                    log_prob = Normal(0, 1).log_prob(z).sum(dim=1) + log_det

                    # Compute the loss (negative log-likelihood)
                    loss = -log_prob.mean()

                    val_loss += loss.item()

                print('Epoch: {} Valid Loss: {:.5f}'.format(epoch,
                                                        val_loss/len(val_loader.dataset)))

                # Save best model according to minima of valid loss
                if val_loss/len(val_loader.dataset) < best_val_loss: # if val loss has decreased
                    best_val_loss = val_loss/len(val_loader.dataset) # update best val loss
                    torch.save(model.state_dict(), f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/best_model.pt")
                    print()
                    print('Best model updated at Epoch {}'.format(epoch))
                    print()

        val_loss_list.append(val_loss/len(val_loader.dataset))

    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/train_loss_list.pkl", 'wb') as file:
        pickle.dump(train_loss_list, file)
    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/val_loss_list.pkl", 'wb') as file:
        pickle.dump(val_loss_list, file)

if __name__ == '__main__':
    model = ConditionalNormalizingFlow(**configs['model_params'])
    train_nf(model, configs)
