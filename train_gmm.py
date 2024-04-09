import os
import sys
import json
import pickle
import torch
import data.utils as utils
sys.modules['utils'] = utils # Way to get around relative imports in utils for ZeoSynGen_dataset # https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
from torch_geometric.loader import DataLoader
from cde.density_simulation import SkewNormal
from cde.density_estimator import KernelMixtureNetwork, LSConditionalDensityEstimation, MixtureDensityNetwork, ConditionalKernelDensityEstimation
import numpy as np
import pdb

configs = { 
            'model_type' : 'gmm',
            'split' : 'system',
            'fname': 'v0',
            # 'device' : 'cuda:3',
            # 'beta' : 1e-2, # optimal 1e-2,
            # 'batch_size' : 2048,
            # 'n_epochs' : 5000,
            # 'lr' : 1e-4,
            'model_params':{
                        'n_centers': 50,
                        'syn_dims': 12, 
                        'zeo_feat_dims': 143, 
                        'osda_feat_dims': 14,
                        },
            }

def train_gmm(model, configs):

    # # Create run folder
    # assert os.path.isdir(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}") == False, 'Name already taken. Please choose another folder name.'
    # os.mkdir(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}")

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
    # pdb.set_trace()

# """ simulate some data """
# density_simulator = SkewNormal(random_seed=22)
# X, Y = density_simulator.simulate(n_samples=3000)

# """ fit density model """
# model = LSConditionalDensityEstimation(ndim_x=1, ndim_y=1, n_centers=50,random_seed=22) # works
# model.fit(X, Y)


# """ query the conditional pdf and cdf """
# x_cond = np.zeros((1, 1))
# y_query = np.ones((1, 1)) * 0.1
# prob = model.pdf(x_cond, y_query)
# pdb.set_trace()

# cum_prob = model.cdf(x_cond, y_query)

# """ compute conditional moments & VaR  """
# mean = model.mean_(x_cond)[0][0]
# std = model.std_(x_cond)[0][0]
# skewness = model.skewness(x_cond)[0]

if __name__ == '__main__':
    model = LSConditionalDensityEstimation(ndim_x=configs['model_params']['zeo_feat_dims']+configs['model_params']['osda_feat_dims'], ndim_y=configs['model_params']['syn_dims'], n_centers=configs['model_params']['n_centers'],random_seed=0)
    train_gmm(model, configs)