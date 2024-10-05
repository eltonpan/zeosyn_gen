import os
import sys
import json
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import torch
from einops import repeat

from eval import load_model

model_type = 'diff'
fname = 'v3'
split = 'system'
cond_scale = 0.75
model, configs = load_model(model_type, fname, split)

# Load test set
with open(f'data/ZeoSynGen_dataset.pkl', 'rb') as f: # load dataset
    dataset = pickle.load(f)
_, _, _dataset = dataset.train_val_test_split(mode=configs['split'], both_graph_feat_present=True, random_state=0) # Note, here we filter out points with no graph/feature present for either zeolite and OSDA

zeo_code, zeo, osda_smiles, osda, = _dataset[3], _dataset[5], _dataset[13], _dataset[15]

print(f"Predicting using diffusion model with cond_scale of {cond_scale}")
n_samples = 15
zeo_code, osda_smiles = repeat(np.array(zeo_code), 'n -> (repeat n)', repeat=n_samples), repeat(np.array(osda_smiles), 'n -> (repeat n)', repeat=n_samples)
zeo, osda = repeat(zeo, 'n d -> (repeat n) d', repeat=n_samples), repeat(osda, 'n d -> (repeat n) d', repeat=n_samples)
zeo, osda = zeo.to(configs['device']), osda.to(configs['device'])
sampled_data = model.sample(batch_size=len(zeo), zeo=zeo, osda=osda, cond_scale=cond_scale, save_trajectory=True)
