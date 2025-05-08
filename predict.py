import os
import numpy as np
import pandas as pd
import pickle
import torch
from einops import repeat
from data.syn_variables import zeo_cols, osda_cols
from data.iza_codes import codes
from eval import load_model
import pdb
import seaborn as sns
import matplotlib.pyplot as plt

# Zeolite and OSDA
zeo, osda = 'UFI', 'C1COCCN2CCOCCOCCN(CCO1)CCOCCOCC2'

# Set model parameters
model_type = 'diff'
fname = 'v3'
cond_scale = 0.75
split = 'system'

def predict(zeo, osda, model_type, fname, cond_scale, split):
    assert os.path.isdir(f"predictions/{zeo}_{osda}") == False, 'Prediction already exists.'

    # Load model
    model, configs = load_model(model_type, fname, split)

    # Load zeolite and OSDA descriptors
    df_zeo = pd.read_csv('data/zeolite_descriptors.csv').drop(columns = ['Unnamed: 0'])
    # df_osda = pd.read_csv('data/osda_descriptors.csv').drop(columns = ['Unnamed: 0'])
    df_osda = pd.read_csv('data/2024-10-02_K222_and_CHA_OSDA_features.csv').drop(columns = ['Unnamed: 0'])

    df_zeo = df_zeo[['Code']+zeo_cols]
    df_osda = df_osda[['osda smiles']+[col for col in osda_cols.keys()]]

    assert zeo in list(df_zeo['Code']), 'Zeolite features not available.'
    assert osda in list(df_osda['osda smiles']), 'OSDA features not available.'

    # print('osda_cols', [x.split('_mean')[0] if '_mean' in x else x for x in osda_cols.keys()])

    zeo_feat = np.array(df_zeo[df_zeo['Code'] == zeo][zeo_cols])
    osda_feat = np.array(df_osda[df_osda['osda smiles'] == osda][list(osda_cols.keys())])
    # print(zeo_feat)
    # print(osda_feat)

    # Scale features
    with open(f'data/scalers/zeo_feat_scaler.pkl', 'rb') as f: # load scaler
        zeo_scaler = pickle.load(f)
    with open(f'data/scalers/osda_feat_scaler.pkl', 'rb') as f: # load scaler
        osda_scaler = pickle.load(f)
    zeo_feat_scaled = torch.tensor(zeo_scaler.transform(zeo_feat), dtype=torch.float32)
    osda_feat_scaled = torch.tensor(osda_scaler.transform(osda_feat), dtype=torch.float32)

    # print(zeo_feat_scaled)
    # print(osda_feat_scaled)

    if configs['model_type'] == 'diff':
        n_samples = 1000
        zeo_feat_scaled, osda_feat_scaled = repeat(zeo_feat_scaled, 'n d -> (repeat n) d', repeat=n_samples), repeat(osda_feat_scaled, 'n d -> (repeat n) d', repeat=n_samples)
        zeo_feat_scaled, osda_feat_scaled = zeo_feat_scaled.to(configs['device']), osda_feat_scaled.to(configs['device'])
        print(zeo_feat_scaled.shape)
        print(osda_feat_scaled.shape)
        
        sampled_data = model.sample(batch_size=n_samples, zeo=zeo_feat_scaled, osda=osda_feat_scaled, cond_scale=cond_scale)
        syn_pred = torch.tensor(sampled_data.squeeze().detach().cpu().numpy())

    # Load dataset (to get quantile transformers)
    with open(f'data/ZeoSynGen_dataset.pkl', 'rb') as f: # load dataset
        dataset = pickle.load(f)

    # Scale synthesis conditions back
    if configs['model_type'] != 'random':
        for ratio_idx, ratio in enumerate(dataset.ratio_names+dataset.cond_names):
            qt = dataset.qts[ratio] # load quantile transformer
            syn_pred[:,ratio_idx] = torch.tensor(qt.inverse_transform(syn_pred[:,ratio_idx].reshape(-1, 1))).reshape(-1) # transform back
    syn_pred = pd.DataFrame(syn_pred, columns=dataset.ratio_names+dataset.cond_names)

    # Save predictions
    os.mkdir(f"predictions/{zeo}_{osda}")
    syn_pred.to_csv(f'predictions/{zeo}_{osda}/syn_pred.csv', index=False)
    print(f"Predictions saved to predictions/{zeo}_{osda}/syn_pred.csv")


if __name__ == '__main__':
    predict(zeo, osda, model_type, fname, cond_scale, split)