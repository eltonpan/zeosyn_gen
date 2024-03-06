import os
import sys
import pdb
import pickle
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import data.utils as utils
sys.modules['utils'] = utils # Way to get around relative imports in utils for ZeoSynGen_dataset # https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from models.cvae import CVAEv1, CVAEv2

configs = { 
            'model_type' : 'cvae',
            'split' : 'system',
            'fname': 'v15',
            'device' : 'cuda:3',
            'beta' : 1e-2, # optimal 1e-2,
            'batch_size' : 2048,
            'n_epochs' : 5000,
            'lr' : 1e-4,
            'model_params':{
                        'z_dims': 2,
                        'zeo_h_dims': 64, 
                        'osda_h_dims': 64, 
                        'syn_dims': 12, 
                        'zeo_feat_dims': 143, 
                        'osda_feat_dims': 14
                        },
            }

def train_cvae(model, configs):

    # Create run folder
    assert os.path.isdir(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}") == False, 'Name already taken. Please choose another folder name.'
    os.mkdir(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}")

    # Save configs
    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/configs.json", "w") as outfile:
        json.dump(configs, outfile, indent=4)

    model = model.to(configs['device'])

    with open(f'data/ZeoSynGen_dataset.pkl', 'rb') as f: # load dataset
        dataset = pickle.load(f)

    train_dataset, val_dataset, test_dataset = dataset.train_val_test_split(mode=configs['split'], both_graph_feat_present=True, random_state=0) # Note, here we filter out points with no graph/feature present for either zeolite and OSDA

    train_dataset = (train_dataset[1], train_dataset[5], train_dataset[15])
    val_dataset   = (val_dataset[1],   val_dataset[5],   val_dataset[15])
    test_dataset  = (test_dataset[1],  test_dataset[5],  test_dataset[15])

    train_loader = DataLoader(list(zip(*train_dataset)), batch_size=2048, shuffle=True)
    val_loader  = DataLoader(list(zip(*val_dataset)),  batch_size=2048, shuffle=False)

    best_model = None # Model to be outputted based on val loss
    train_loss_list  = []
    recons_loss_list = [] 
    kld_loss_list    = []
    val_loss_list    = []
    
    best_val_loss = 1e10
    optimizer = optim.Adam(model.parameters(), lr = configs['lr'])
    
    for epoch in tqdm(range(configs['n_epochs'])): 

        # Training
        model.train()
        train_loss        = 0.
        recons_loss_total = 0.
        kld_loss_total    = 0.

        for (syn, zeo, osda) in train_loader:

            syn = syn.to(configs['device'])
            zeo = zeo.to(configs['device'])
            osda = osda.to(configs['device'])

            optimizer.zero_grad()
            syn_prime, mu, logvar = model(syn, zeo, osda)

            # Calculate loss
            recons_loss = F.mse_loss(syn_prime, syn, reduction='sum')/2.0
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recons_loss + configs['beta']*kld_loss

            loss.backward()
            train_loss        += loss.item()
            recons_loss_total += recons_loss.item()
            kld_loss_total    += kld_loss.item()
            
            optimizer.step()
        
        print('Epoch: {} Recon Loss: {:.5f} KLD Loss: {:.5f} Train Loss: {:.5f}'.format(
        epoch, 
        recons_loss_total / len(train_loader.dataset),
        kld_loss_total    / len(train_loader.dataset),
        train_loss        / len(train_loader.dataset), ))
        
        train_loss_list.append(train_loss        / len(train_loader.dataset))
        recons_loss_list.append(recons_loss_total / len(train_loader.dataset))
        kld_loss_list.append(kld_loss_total    / len(train_loader.dataset))
        
        # Validation
        if epoch%250 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.
                for (syn, zeo, osda) in val_loader:
                    syn = syn.to(configs['device'])
                    zeo = zeo.to(configs['device'])
                    osda = osda.to(configs['device'])
                    syn_prime, mu, logvar = model(syn, zeo, osda)

                    # Calculate loss
                    recons_loss = F.mse_loss(syn_prime, syn, reduction='sum')/2.0
                    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recons_loss + configs['beta']*kld_loss
                    
                    val_loss += loss.item()
                
                print('Epoch: {} Valid Loss: {:.5f}'.format(epoch,
                                                        val_loss/len(val_loader.dataset)))
                    
                # Save best model according to minima of valid loss
                if val_loss/len(val_loader.dataset) < best_val_loss: # if val loss has decreased
                    best_val_loss = val_loss/len(val_loader.dataset) # update best val loss
                    best_model = model # update model
                    print()
                    print('Best model updated at Epoch {}'.format(epoch))
                    torch.save(model.state_dict(), f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/best_model.pt")
                    print()
                    
        val_loss_list.append(val_loss / len(val_loader.dataset))
    

    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/train_loss_list.pkl", 'wb') as file:
        pickle.dump(train_loss_list, file)
    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/recons_loss_list.pkl", 'wb') as file:
        pickle.dump(recons_loss_list, file)
    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/kld_loss_list.pkl", 'wb') as file:
        pickle.dump(kld_loss_list, file)
    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/val_loss_list.pkl", 'wb') as file:
        pickle.dump(val_loss_list, file)

    # return best_model, train_loss_list, recons_loss_list, kld_loss_list, val_loss_list

if __name__ == "__main__":
    model = CVAEv2(**configs['model_params'])
    train_cvae(model, configs)