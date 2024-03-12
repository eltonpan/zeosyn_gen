import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
import json
import data.utils as utils
sys.modules['utils'] = utils # Way to get around relative imports in utils for ZeoSynGen_dataset # https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
from einops import repeat
from models.cvae import CVAEv1, CVAEv2
from data.metrics import maximum_mean_discrepancy, wasserstein_distance
import pdb

def load_model(model_type, fname, split):
    '''
    Load model and configs
    '''
    print('Loading model and configs...')
    # 1) Load configs
    with open(f'runs/{model_type}/{split}/{fname}/configs.json') as f:
        configs = json.load(f)

    # 2) Load model and get predictions
    # model = CVAEv1(**configs['model_params'])
    model = CVAEv2(**configs['model_params'])
    model.load_state_dict(torch.load(f'runs/{model_type}/{split}/{fname}/best_model.pt', map_location=configs['device']))
    model = model.to(configs['device'])
    model.eval()

    return model, configs

def get_prediction_and_ground_truths(model, configs):
    # '''
    # Get predicted distributions (and their scaled versions) and ground truth distributions for synthesis conditions
    # '''
    print('Getting model predictions and grouth truths...')
    # Load test set
    with open(f'data/ZeoSynGen_dataset.pkl', 'rb') as f: # load dataset
        dataset = pickle.load(f)
    _, _, test_dataset = dataset.train_val_test_split(mode=configs['split'], both_graph_feat_present=True, random_state=0) # Note, here we filter out points with no graph/feature present for either zeolite and OSDA

    # Get test zeolites and OSDAs
    zeo_code, zeo, osda_smiles, osda, = test_dataset[3], test_dataset[5], test_dataset[13], test_dataset[15], 

    if not os.path.isfile(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/syn_pred_agg.csv"): # If synthetic predictions not already saved
        print('Systems not predicted yet, predicting and saving...')
        
        # Predict synthesis conditions
        zeo_code = repeat(np.array(zeo_code), 'n -> (repeat n)', repeat=50)
        zeo = repeat(zeo, 'n d -> (repeat n) d', repeat=50)
        osda_smiles = repeat(np.array(osda_smiles), 'n -> (repeat n)', repeat=50)
        osda = repeat(osda, 'n d -> (repeat n) d', repeat=50)
        zeo, osda = zeo.to(configs['device']), osda.to(configs['device'])
        syn_pred = torch.tensor(model.predict(zeo, osda).cpu().detach().numpy())

        # Scale synthesis conditions back
        for ratio_idx, ratio in enumerate(dataset.ratio_names+dataset.cond_names):
            qt = dataset.qts[ratio] # load quantile transformer
            syn_pred[:,ratio_idx] = torch.tensor(qt.inverse_transform(syn_pred[:,ratio_idx].reshape(-1, 1))).reshape(-1) # transform back
        syn_pred = pd.DataFrame(syn_pred, columns=dataset.ratio_names+dataset.cond_names)
        syn_pred['zeo'], syn_pred['osda'] = zeo_code, osda_smiles
        syn_pred.to_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/syn_pred_agg.csv", index=False) # Save synthetic predictions
        
    else:
        print('Loading synthetic predictions from saved predictions...')
        syn_pred = pd.read_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/syn_pred_agg.csv")

    syn_pred_scaled = utils.scale_x_syn_ratio(syn_pred, dataset) # get min-max scaled version too

    # Get ground truth
    syn_true, zeo_code, osda_smiles = test_dataset[1], test_dataset[3], test_dataset[13]
    for ratio_idx, ratio in enumerate(dataset.ratio_names+dataset.cond_names):
        qt = dataset.qts[ratio] # load quantile transformer
        syn_true[:,ratio_idx] = torch.tensor(qt.inverse_transform(syn_true[:,ratio_idx].reshape(-1, 1))).reshape(-1) # transform back
    syn_true = pd.DataFrame(syn_true, columns=dataset.ratio_names+dataset.cond_names)
    syn_true = pd.DataFrame(syn_true, columns=dataset.ratio_names+dataset.cond_names)
    syn_true['zeo'], syn_true['osda'] = zeo_code, osda_smiles

    syn_true_scaled = utils.scale_x_syn_ratio(syn_true, dataset) # get min-max scaled version too

    return syn_pred, syn_pred_scaled, syn_true, syn_true_scaled, dataset

def eval_zeolite_aggregated(syn_pred, syn_pred_scaled, syn_true, syn_true_scaled, dataset, configs, eval=True, plot=False, print_metrics=False, num_systems=None):
    '''
    Calculate and save evaluation metrics for zeolite-aggregated systems

    Args:
    eval: bool, whether to calculate metrics
    plot: bool, whether to plot prediction vs. true
    print_metrics: bool, whether to print metrics
    num_systems: int, number of systems (in descending frequency in test dataset)
    '''

    if print_metrics:
        assert eval, 'If printing metrics, eval must be True'
    
    print('Calculating metrics for zeolite-aggregated systems...')
    # 4) Evaluate on zeolite-aggregated systems
    zeo_systems = list(syn_true['zeo'].value_counts().index)
    count = 0
    mmd_zeo_agg = {} # Dict of MMDs for each zeolite
    wsd_zeo_agg = {} # Dict of WSDs for each zeolite
    for zeo in zeo_systems:
        if zeo != 'Dense/Amorphous':
            if plot:
                print(zeo)

            sys_syn_pred, sys_syn_true = syn_pred[syn_pred['zeo'] == zeo], syn_true[syn_true['zeo'] == zeo]
            sys_syn_pred_scaled, sys_syn_true_scaled = syn_pred_scaled[syn_pred_scaled['zeo'] == zeo], syn_true_scaled[syn_true_scaled['zeo'] == zeo]
            
            if eval:
                # MMD
                mmd = maximum_mean_discrepancy(sys_syn_pred_scaled[dataset.ratio_names+dataset.cond_names], sys_syn_true_scaled[dataset.ratio_names+dataset.cond_names])
                mmd_zeo_agg[zeo] = mmd

                # WSD
                wsd = wasserstein_distance(sys_syn_pred_scaled[dataset.ratio_names+dataset.cond_names], sys_syn_true_scaled[dataset.ratio_names+dataset.cond_names])
                wsd_zeo_agg[zeo] = wsd

            if print_metrics:
                print('MMD:', mmd)
                print('WSD:', wsd)

            # Plot prediction vs. true
            if plot:
                utils.compare_gel_conds([sys_syn_pred, sys_syn_true], ['Predicted', 'True'], [True, False], [False, True], ['tab:orange', 'tab:blue'], common_norm=True, alpha=0.5)
            
            count += 1

        if count == num_systems:
            break
    
    if eval:
        # Save MMD
        mmd_zeo_agg_df = pd.DataFrame({'zeo': mmd_zeo_agg.keys(), 'MMD': mmd_zeo_agg.values()})
        assert mmd_zeo_agg_df['MMD'].isna().sum() == 0 # Check no NaNs
        mmd_mean, mmd_std = mmd_zeo_agg_df['MMD'].mean(), mmd_zeo_agg_df['MMD'].std()
        print('Mean MMD:', mmd_mean, 'Std MMD:', mmd_std)
        if num_systems == None: # Save only if evaluated on all systems
            mmd_zeo_agg_df.to_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/mmd_zeo_agg_df.csv", index=False) # Save per-system MMDs
            with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/mmd_zeo_agg.json", "w") as outfile:
                json.dump({'MMD mean': mmd_mean, 'MMD std': mmd_std}, outfile, indent=4)

        # Save WSD
        wsd_zeo_agg_df = pd.DataFrame({'zeo': wsd_zeo_agg.keys(), 'WSD': wsd_zeo_agg.values()})
        assert wsd_zeo_agg_df['WSD'].isna().sum() == 0 # Check no NaNs
        wsd_mean, wsd_std = wsd_zeo_agg_df['WSD'].mean(), wsd_zeo_agg_df['WSD'].std()
        print('Mean WSD:', wsd_mean, 'Std WSD:', wsd_std)
        if num_systems == None: # Save only if evaluated on all systems
            wsd_zeo_agg_df.to_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/wsd_zeo_agg_df.csv", index=False) # Save per-system WSDs
            with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/wsd_zeo_agg.json", "w") as outfile:
                json.dump({'WSD mean': wsd_mean, 'WSD std': wsd_std}, outfile, indent=4)
        
        return mmd_zeo_agg_df, wsd_zeo_agg_df
    
    else:
        pass

def eval_zeolite_osda(syn_pred, syn_pred_scaled, syn_true, syn_true_scaled, dataset, configs, eval=True, plot=False, print_metrics=False, num_systems=None):
    '''
    Calculate and save evaluation metrics for zeolite-OSDA systems

    Args:
    eval: bool, whether to calculate metrics
    plot: bool, whether to plot prediction vs. true
    print_metrics: bool, whether to print metrics
    num_systems: int, number of systems (in descending frequency in test dataset)
    '''

    if print_metrics:
        assert eval, 'If printing metrics, eval must be True'

    print('Calculating metrics for zeolite-OSDA systems...')
    # 5) Evaluate on zeolite-osda systems
    zeo_osda_systems = list(syn_true[['zeo', 'osda']].value_counts().index)
    count = 0
    mmd_zeo_osda = {} # Dict of MMDs for each zeolite-osda 
    wsd_zeo_osda = {} # Dict of WSDs for each zeolite-osda 
    for zeo, osda in zeo_osda_systems:
        if zeo != 'Dense/Amorphous':
            if plot:
                print(zeo, osda)
            sys_syn_pred, sys_syn_true = syn_pred[(syn_pred['zeo'] == zeo) & (syn_pred['osda'] == osda)], syn_true[(syn_true['zeo'] == zeo) & (syn_true['osda'] == osda)]
            sys_syn_pred_scaled, sys_syn_true_scaled = syn_pred_scaled[(syn_pred_scaled['zeo'] == zeo) & (syn_pred_scaled['osda'] == osda)], syn_true_scaled[(syn_true_scaled['zeo'] == zeo) & (syn_true_scaled['osda'] == osda)]

            if eval:
                # MMD
                mmd = maximum_mean_discrepancy(sys_syn_pred_scaled[dataset.ratio_names+dataset.cond_names], sys_syn_true_scaled[dataset.ratio_names+dataset.cond_names])
                mmd_zeo_osda[(zeo,osda)] = mmd

                # WSD
                wsd = wasserstein_distance(sys_syn_pred_scaled[dataset.ratio_names+dataset.cond_names], sys_syn_true_scaled[dataset.ratio_names+dataset.cond_names])
                wsd_zeo_osda[(zeo,osda)] = wsd

            if print_metrics:
                print('MMD:', mmd)
                print('WSD:', wsd)

            if plot:
                utils.compare_gel_conds([sys_syn_pred, sys_syn_true], ['Predicted', 'True'], [True, False], [False, True], ['tab:orange', 'tab:blue'], common_norm=True, alpha=0.5)

            count += 1
            
        if count == num_systems:
            break
    if eval:
        # Save MMD
        mmd_zeo_osda_df = pd.DataFrame({'zeo': [z for z, _ in [*mmd_zeo_osda]], 'osda': [o for _, o in [*mmd_zeo_osda]], 'MMD': mmd_zeo_osda.values()})
        assert mmd_zeo_osda_df['MMD'].isna().sum() == 0 # Check no NaNs
        mmd_mean, mmd_std = mmd_zeo_osda_df['MMD'].mean(), mmd_zeo_osda_df['MMD'].std()
        print('Mean MMD:', mmd_mean, 'Std MMD:', mmd_std)
        if num_systems == None: # Save only if evaluated on all systems
            mmd_zeo_osda_df.to_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/mmd_zeo_osda_df.csv", index=False) # Save per-system MMDs
            with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/mmd_zeo_osda.json", "w") as outfile:
                json.dump({'MMD mean': mmd_mean, 'MMD std': mmd_std}, outfile, indent=4)

        # Save WSD
        wsd_zeo_osda_df = pd.DataFrame({'zeo': [z for z, _ in [*wsd_zeo_osda]], 'osda': [o for _, o in [*wsd_zeo_osda]], 'WSD': wsd_zeo_osda.values()})
        assert wsd_zeo_osda_df['WSD'].isna().sum() == 0 # Check no NaNs
        wsd_mean, wsd_std = wsd_zeo_osda_df['WSD'].mean(), wsd_zeo_osda_df['WSD'].std()
        print('Mean WSD:', wsd_mean, 'Std WSD:', wsd_std)
        if num_systems == None: # Save only if evaluated on all systems
            wsd_zeo_osda_df.to_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/wsd_zeo_osda_df.csv", index=False) # Save per-system WSDs
            with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/wsd_zeo_osda.json", "w") as outfile:
                json.dump({'WSD mean': wsd_mean, 'WSD std': wsd_std}, outfile, indent=4)

        return mmd_zeo_osda_df, wsd_zeo_osda_df

    else:
        pass
    
def eval_single_system(syn_pred, syn_pred_scaled, syn_true, syn_true_scaled, dataset, mmd_zeo_agg_df, wsd_zeo_agg_df, mmd_zeo_osda_df, wsd_zeo_osda_df, zeo, osda=None, plot=True, print_metrics=True):
    '''
    Calculate evaluation metrics for a SPECIFIC zeolite or zeolite-OSDA system

    Args:
    eval: bool, whether to calculate metrics
    plot: bool, whether to plot prediction vs. true
    print_metrics: bool, whether to print metrics
    '''

    if osda == None: # Zeolite-aggregated
        if plot:
            print(zeo)

        sys_syn_pred, sys_syn_true = syn_pred[syn_pred['zeo'] == zeo], syn_true[syn_true['zeo'] == zeo]
        sys_syn_pred_scaled, sys_syn_true_scaled = syn_pred_scaled[syn_pred_scaled['zeo'] == zeo], syn_true_scaled[syn_true_scaled['zeo'] == zeo]

        if print_metrics:
            mmd = mmd_zeo_agg_df[mmd_zeo_agg_df['zeo']==zeo]['MMD'].item()
            wsd = wsd_zeo_agg_df[wsd_zeo_agg_df['zeo']==zeo]['WSD'].item()
            print('MMD:', mmd)
            print('WSD:', wsd)

    else: # Zeolite-OSDA
        if plot:
            print(zeo, osda)

        sys_syn_pred, sys_syn_true = syn_pred[(syn_pred['zeo'] == zeo) & (syn_pred['osda'] == osda)], syn_true[(syn_true['zeo'] == zeo) & (syn_true['osda'] == osda)]
        sys_syn_pred_scaled, sys_syn_true_scaled = syn_pred_scaled[(syn_pred_scaled['zeo'] == zeo) & (syn_pred_scaled['osda'] == osda)], syn_true_scaled[(syn_true_scaled['zeo'] == zeo) & (syn_true_scaled['osda'] == osda)]
        
        if print_metrics:
            mmd = mmd_zeo_osda_df[(mmd_zeo_osda_df['zeo']==zeo) & (mmd_zeo_osda_df['osda']==osda)]['MMD'].item()
            wsd = wsd_zeo_osda_df[(wsd_zeo_osda_df['zeo']==zeo) & (wsd_zeo_osda_df['osda']==osda)]['WSD'].item()
            print('MMD:', mmd)
            print('WSD:', wsd)

    if plot:
        utils.compare_gel_conds([sys_syn_pred, sys_syn_true], ['Predicted', 'True'], [True, False], [False, True], ['tab:orange', 'tab:blue'], common_norm=True, alpha=0.5)

def get_metric_dataframes(configs):
    '''
    Get dataframes of calculated metrics
    '''

    assert os.path.isfile(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/mmd_zeo_agg_df.csv"), 'Full evaluation dataframe (mmd_zeo_agg_df.csv) must exist, else run eval_cvae.py to get it'
    mmd_zeo_agg_df = pd.read_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/mmd_zeo_agg_df.csv")

    assert os.path.isfile(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/wsd_zeo_agg_df.csv"), 'Full evaluation dataframe (wsd_zeo_agg_df.csv) must exist, else run eval_cvae.py to get it'
    wsd_zeo_agg_df = pd.read_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/wsd_zeo_agg_df.csv")

    assert os.path.isfile(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/mmd_zeo_osda_df.csv"), 'Full evaluation dataframe (mmd_zeo_osda_df.csv) must exist, else run eval_cvae.py to get it'
    mmd_zeo_osda_df = pd.read_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/mmd_zeo_osda_df.csv")

    assert os.path.isfile(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/wsd_zeo_osda_df.csv"), 'Full evaluation dataframe (wsd_zeo_osda_df.csv) must exist, else run eval_cvae.py to get it'
    wsd_zeo_osda_df = pd.read_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/wsd_zeo_osda_df.csv")

    return mmd_zeo_agg_df, wsd_zeo_agg_df, mmd_zeo_osda_df, wsd_zeo_osda_df
    
if __name__ == '__main__':
    model_type = 'cvae'
    fname = 'v10'
    split = 'system'

    model, configs = load_model(model_type, fname, split)
    syn_pred, syn_pred_scaled, syn_true, syn_true_scaled, dataset = get_prediction_and_ground_truths(model, configs)
    mmd_zeo_agg_df, wsd_zeo_agg_df = eval_zeolite_aggregated(syn_pred, syn_pred_scaled, syn_true, syn_true_scaled, dataset, configs)
    mmd_zeo_osda_df, wsd_zeo_osda_df = eval_zeolite_osda(syn_pred, syn_pred_scaled, syn_true, syn_true_scaled, dataset, configs)
