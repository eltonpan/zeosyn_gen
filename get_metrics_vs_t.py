import os
import sys
import json
import pickle
import matplotlib.pyplot as plt
import pandas as pd

import torch
import data.utils as utils
sys.modules['utils'] = utils # Way to get around relative imports in utils for ZeoSynGen_dataset # https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
from eval import load_model, get_prediction_and_ground_truths, eval_zeolite_aggregated, eval_zeolite_osda, plot_single_system
from data.metrics import maximum_mean_discrepancy, wasserstein_distance
from models.diffusion import *
import tqdm
import pdb

# This code block is needed to get the zeolite and OSDA labels
model_type = 'diff'
fname = 'v3'
split = 'system'
cond_scale = 0.75

with open(f'data/ZeoSynGen_dataset.pkl', 'rb') as f: # load dataset
    dataset = pickle.load(f)
_, _, _dataset = dataset.train_val_test_split(mode='system', both_graph_feat_present=True, random_state=0) # Note, here we filter out points with no graph/feature present for either zeolite and OSDA

zeo_code, zeo, osda_smiles, osda, = _dataset[3], _dataset[5], _dataset[13], _dataset[15]

print(f"Predicting using diffusion model with cond_scale of {cond_scale}")
n_samples = 15
zeo_code, osda_smiles = repeat(np.array(zeo_code), 'n -> (repeat n)', repeat=n_samples), repeat(np.array(osda_smiles), 'n -> (repeat n)', repeat=n_samples)

print(len(zeo_code), len(osda_smiles))

# Get ground truth
syn_true, zeo_code_true, osda_smiles_true = _dataset[1], _dataset[3], _dataset[13]
for ratio_idx, ratio in enumerate(dataset.ratio_names+dataset.cond_names):
    qt = dataset.qts[ratio] # load quantile transformer
    syn_true[:,ratio_idx] = torch.tensor(qt.inverse_transform(syn_true[:,ratio_idx].reshape(-1, 1))).reshape(-1) # transform back
syn_true = pd.DataFrame(syn_true, columns=dataset.ratio_names+dataset.cond_names)
syn_true['zeo'], syn_true['osda'] = zeo_code_true, osda_smiles_true
syn_true_scaled = utils.scale_x_syn_ratio(syn_true, dataset) # get min-max scaled version too
syn_true

ts = list(reversed(range(0, 1000, 1)))
wsds = []

for t in tqdm.tqdm(ts):
    print(t)
    if (not os.path.isfile(f"data/diffusion_trajectory/wsd_vs_t/wsd_t{t}.pkl")) or (not os.path.isfile(f"data/diffusion_trajectory/f1_vs_t/f1_t{t}.csv")) or (not os.path.isfile(f"data/diffusion_trajectory/prec_vs_t/prec_t{t}.csv")) or (not os.path.isfile(f"data/diffusion_trajectory/rec_vs_t/rec_t{t}.csv")):
        with open(f'data/diffusion_trajectory/t{t}.pkl', 'rb') as file:
            syn_pred = pickle.load(file)
            assert len(syn_pred) == len(zeo_code) == len(osda_smiles)

            syn_pred = syn_pred.squeeze()

            # Scale synthesis conditions back
            for ratio_idx, ratio in enumerate(dataset.ratio_names+dataset.cond_names):
                qt = dataset.qts[ratio] # load quantile transformer
                syn_pred[:,ratio_idx] = torch.tensor(qt.inverse_transform(syn_pred[:,ratio_idx].reshape(-1, 1))).reshape(-1) # transform back

        syn_pred = pd.DataFrame(syn_pred, columns=dataset.ratio_names+dataset.cond_names)
        syn_pred['zeo'], syn_pred['osda'] = zeo_code, osda_smiles
        syn_pred_scaled = utils.scale_x_syn_ratio(syn_pred, dataset)

        # Calculate metrics
        mmd_zeo_osda_df, wsd_zeo_osda_df, prec_zeo_osda_df_mean, rec_zeo_osda_df_mean, f1_zeo_osda_df_mean = eval_zeolite_osda(syn_pred, syn_pred_scaled, syn_true, syn_true_scaled, dataset, {
                                                                                                        "model_type": model_type,
                                                                                                        "split": split,
                                                                                                        "fname": fname,
                                                                                                        "device": "cuda:1",},
                                                                                                        return_f1=True,
                                                                                                        )

        wsd_mean =  wsd_zeo_osda_df['WSD'].mean()

        # Save metrics
        with open(f"data/diffusion_trajectory/wsd_vs_t/wsd_t{t}.pkl", 'wb') as file:
            pickle.dump(wsd_mean, file)
        prec_zeo_osda_df_mean.to_csv(f"data/diffusion_trajectory/prec_vs_t/prec_t{t}.csv")
        rec_zeo_osda_df_mean.to_csv(f"data/diffusion_trajectory/rec_vs_t/rec_t{t}.csv")
        f1_zeo_osda_df_mean.to_csv(f"data/diffusion_trajectory/f1_vs_t/f1_t{t}.csv")
        
    else:
        print('Metric already calculated. Loading from file.')

        # Load metrics
        with open(f"data/diffusion_trajectory/wsd_vs_t/wsd_t{t}.pkl", 'rb') as file:
            wsd_mean = pickle.load(file)
        prec_zeo_osda_df_mean = pd.read_csv(f"data/diffusion_trajectory/prec_vs_t/prec_t{t}.csv")
        rec_zeo_osda_df_mean = pd.read_csv(f"data/diffusion_trajectory/rec_vs_t/rec_t{t}.csv")
        f1_zeo_osda_df_mean = pd.read_csv(f"data/diffusion_trajectory/f1_vs_t/f1_t{t}.csv")
    
    print(wsd_mean)
    print(prec_zeo_osda_df_mean)
    print(rec_zeo_osda_df_mean)
    print(f1_zeo_osda_df_mean)

    wsds.append(wsd_mean)

wsds