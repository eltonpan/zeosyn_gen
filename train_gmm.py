import os
import sys
import json
import pickle
import torch
import data.utils as utils
sys.modules['utils'] = utils # Way to get around relative imports in utils for ZeoSynGen_dataset # https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
from torch_geometric.loader import DataLoader
from cde.density_estimator import KernelMixtureNetwork, LSConditionalDensityEstimation, MixtureDensityNetwork, ConditionalKernelDensityEstimation
import numpy as np
import pdb

configs = { 
            'model_type' : 'gmm',
            'split' : 'system',
            'fname': 'v0',
            'model_params':{
                        'n_centers': 50,
                        'syn_dims': 12, 
                        'zeo_feat_dims': 143, 
                        'osda_feat_dims': 14,
                        },
            }

def train_gmm(model, configs):

    # Create run folder
    assert os.path.isdir(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}") == False, 'Name already taken. Please choose another folder name.'
    os.mkdir(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}")

    # Save configs
    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/configs.json", "w") as outfile:
        json.dump(configs, outfile, indent=4)

    with open(f'data/ZeoSynGen_dataset.pkl', 'rb') as f: # load dataset
        dataset = pickle.load(f)
    
    train_dataset, val_dataset, _ = dataset.train_val_test_split(mode=configs['split'], both_graph_feat_present=True, random_state=0) # Note, here we filter out points with no graph/feature present for either zeolite and OSDA

    syn_train, zeo_train, osda_train = train_dataset[1], train_dataset[5], train_dataset[15]
    syn_val, zeo_val, osda_val = val_dataset[1],   val_dataset[5],   val_dataset[15]

    X = torch.cat([zeo_train, osda_train], dim=1).numpy()
    Y = syn_train.numpy()

    model.fit(X, Y, verbose=True)

    # Save model
    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/model.pkl", "wb") as outfile:
        pickle.dump(model, outfile)

if __name__ == '__main__':
    model = LSConditionalDensityEstimation(ndim_x=configs['model_params']['zeo_feat_dims']+configs['model_params']['osda_feat_dims'], ndim_y=configs['model_params']['syn_dims'], n_centers=configs['model_params']['n_centers'],random_seed=0)
    train_gmm(model, configs)