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
from models.cvae import CVAEv2, CVAE_EQ, CVAE_GNN
from models.gan import Generator
from models.nf import ConditionalNormalizingFlow
from models.bnn import BayesLinear
from models.diffusion import *
from models.nn import NN
from torch_geometric.loader import DataLoader
from data.metrics import maximum_mean_discrepancy, wasserstein_distance, abs_error, coverage
from sklearn.metrics import r2_score
import tqdm
import pdb

def load_model(model_type, fname, split, load_step=None):
    '''
    Load model and configs

    Args:
    load_step: Int. Step at which to load model. If None, loads model with lowest validation loss
    '''

    print('Loading model and configs...')
    # 1) Load configs
    with open(f'runs/{model_type}/{split}/{fname}/configs.json') as f:
        configs = json.load(f)

    # 2) Load model and get predictions
    if model_type == 'cvae':
        # model = CVAEv1(**configs['model_params'])
        model = CVAEv2(**configs['model_params'])
        model.load_state_dict(torch.load(f'runs/{model_type}/{split}/{fname}/best_model.pt', map_location=configs['device']))
        model = model.to(configs['device'])
        model.eval()

    if model_type == 'cvae-eq':
        model = CVAE_EQ(**configs['model_params'])
        model.load_state_dict(torch.load(f'runs/{model_type}/{split}/{fname}/best_model.pt', map_location=configs['device']))
        model = model.to(configs['device'])
        model.eval()
    
    if model_type == 'cvae-gnn':
        model = CVAE_GNN(**configs['model_params'])
        model.load_state_dict(torch.load(f'runs/{model_type}/{split}/{fname}/best_model.pt', map_location=configs['device']))
        model = model.to(configs['device'])
        model.eval()

    elif model_type == 'gan':
        model = Generator(generator_layer_size=configs['model_params']['generator_layer_size'], z_dims=configs['model_params']['z_dims'])
        model.load_state_dict(torch.load(f'runs/{model_type}/{split}/{fname}/gen.pt', map_location=configs['device']))
        model = model.to(configs['device'])
        model.eval()

    elif model_type == 'nf':
        model = ConditionalNormalizingFlow(**configs['model_params'])
        model.load_state_dict(torch.load(f'runs/{model_type}/{split}/{fname}/best_model.pt', map_location=configs['device']))
        model = model.to(configs['device'])
        model.eval()
    
    elif model_type == 'diff':
        if 'save_all_model_checkpoints' not in configs.keys(): # Retro-fit for old configs
            configs['save_all_model_checkpoints'] = False
        if 'dropout' not in configs['model_params'].keys(): # Retro-fit for old configs
            configs['model_params']['dropout'] = False

        unet = Unet1D(
                dim                 = configs['model_params']['dim'],
                dim_mults           = configs['model_params']['dim_mults'],
                channels            = configs['model_params']['channels'],
                resnet_block_groups = configs['model_params']['resnet_block_groups'],
                zeo_feat_dims       = configs['model_params']['zeo_feat_dims'],
                osda_feat_dims      = configs['model_params']['osda_feat_dims'],
                cond_drop_prob      = configs['model_params']['cond_drop_prob'],
                dropout             = configs['model_params']['dropout'],
                )

        model = GaussianDiffusion1D(
                unet,
                seq_length = configs['model_params']['seq_length'],
                timesteps  = configs['model_params']['timesteps'],
                objective  = 'pred_v',
                ).to(configs['device'])
        

        if configs['save_all_model_checkpoints']:
            if load_step == None:
                path = f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}"
                files = [f for f in os.listdir(path) if (".pt" in f)]
                steps = [int(f.split('model_ep')[-1][:-3]) for f in files]
                load_step = max(steps)
            print(f'Loading model at step {load_step}...')
            model.load_state_dict(torch.load(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/model_ep{load_step}.pt", map_location=configs['device']))
        else:
            model.load_state_dict(torch.load(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/model.pt", map_location=configs['device']))
        model.eval()

    elif model_type == 'bnn':
        model = nn.Sequential(BayesLinear(prior_mu=configs['model_params']['prior_mu'], prior_sigma=configs['model_params']['prior_sigma'], in_features=157, out_features=128), 
                          nn.ReLU(), 
                          BayesLinear(prior_mu=configs['model_params']['prior_mu'], prior_sigma=configs['model_params']['prior_sigma'], in_features=128, out_features=12),)
        model.load_state_dict(torch.load(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/best_model.pt", map_location=configs['device']))
        model = model.to(configs['device'])
        model.eval()
    
    elif model_type == 'gmm':
        try: # requires tensorflow. make sure to run with "cde" environment
            with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/model.pkl", 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            print(e)
            model = None
    
    elif model_type in ['nn', 'amd']:
        model = NN(**configs['model_params'])
        model.load_state_dict(torch.load(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/best_model.pt", map_location=configs['device']))
        model = model.to(configs['device'])
        model.eval()
            
    elif model_type == 'random':
        model = None 

    return model, configs

def get_prediction_and_ground_truths(model, configs, cond_scale=None, split='test'):
    '''
    Get predicted distributions (and their scaled versions) and ground truth distributions for synthesis conditions

    Args:
    cond_scale: float. Scale of conditioning for classifier-free guidance. Anything greater than 1 strengthens the classifier-free guidance. reportedly 3-8 is good empirically
    '''

    print('Getting model predictions and grouth truths...')
    # Load test set
    with open(f'data/ZeoSynGen_dataset.pkl', 'rb') as f: # load dataset
        dataset = pickle.load(f)
    if split == 'test':
        _, _, _dataset = dataset.train_val_test_split(mode=configs['split'], both_graph_feat_present=True, random_state=0) # Note, here we filter out points with no graph/feature present for either zeolite and OSDA
    elif split == 'val':
        _, _dataset, _ = dataset.train_val_test_split(mode=configs['split'], both_graph_feat_present=True, random_state=0) # Note, here we filter out points with no graph/feature present for either zeolite and OSDA
    elif split == 'train':
        _dataset, _, _ = dataset.train_val_test_split(mode=configs['split'], both_graph_feat_present=True, random_state=0) # Note, here we filter out points with no graph/feature present for either zeolite and OSDA

    # Get test zeolites and OSDAs
    if configs['model_type'] in ['cvae-eq', 'cvae-gnn']:
        zeo_code, zeo, osda_smiles, osda, = _dataset[3], _dataset[4], _dataset[13], _dataset[15]
    else:
        zeo_code, zeo, osda_smiles, osda, = _dataset[3], _dataset[5], _dataset[13], _dataset[15]

    if configs['model_type'] in ['cvae', 'cvae-eq', 'cvae-gnn', 'gan', 'nf', 'bnn', 'gmm', 'nn', 'random', 'amd']: # prediction csv filename
        pred_fname = "syn_pred_agg.csv"
    elif configs['model_type'] == 'diff':
        assert cond_scale != None, 'cond_scale must be provided for diffusion model'
        pred_fname = f"syn_pred_agg-cond_scale_{cond_scale}-{split}.csv"

    if not os.path.isfile(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/{pred_fname}"): # If synthetic predictions not already saved
        print('Systems not predicted yet, predicting and saving...')
        
        # Predict synthesis conditions
        if configs['model_type'] == 'cvae':
            assert cond_scale == None, 'cond_scale must not be provided for CVAE model'
            zeo_code = repeat(np.array(zeo_code), 'n -> (repeat n)', repeat=50)
            zeo = repeat(zeo, 'n d -> (repeat n) d', repeat=50)
            osda_smiles = repeat(np.array(osda_smiles), 'n -> (repeat n)', repeat=50)
            osda = repeat(osda, 'n d -> (repeat n) d', repeat=50)
            zeo, osda = zeo.to(configs['device']), osda.to(configs['device'])
            syn_pred = torch.tensor(model.predict(zeo, osda).cpu().detach().numpy())

        elif configs['model_type'] == 'cvae-eq':
            assert cond_scale == None, 'cond_scale must not be provided for CVAE model'
            # Batch inference of graphs
            zeo_code = repeat(np.array(zeo_code), 'n -> (repeat n)', repeat=50)
            zeo = zeo*50
            osda_smiles = repeat(np.array(osda_smiles), 'n -> (repeat n)', repeat=50)
            osda = repeat(osda, 'n d -> (repeat n) d', repeat=50)

            dl = DataLoader(list(zip(zeo, osda)), batch_size=2048, shuffle=False) # use dataloader to batch graphs
            syn_pred_list = []
            for (zeo, osda) in tqdm.tqdm(dl):
                zeo, osda = zeo.to(configs['device']), osda.to(configs['device'])
                syn_pred = torch.tensor(model.predict(zeo, osda).cpu().detach().numpy())
                syn_pred_list.append(syn_pred)
            syn_pred = torch.cat(syn_pred_list, dim=0).cpu().detach().numpy()

        elif configs['model_type'] == 'cvae-gnn':
            assert cond_scale == None, 'cond_scale must not be provided for CVAE model'
            # Batch inference of graphs
            zeo_code = repeat(np.array(zeo_code), 'n -> (repeat n)', repeat=50)
            zeo = zeo*50
            osda_smiles = repeat(np.array(osda_smiles), 'n -> (repeat n)', repeat=50)
            osda = repeat(osda, 'n d -> (repeat n) d', repeat=50)

            dl = DataLoader(list(zip(zeo, osda)), batch_size=1024, shuffle=False) # use dataloader to batch graphs
            syn_pred_list = []
            for (zeo, osda) in tqdm.tqdm(dl):
                zeo, osda = zeo.to(configs['device']), osda.to(configs['device'])
                syn_pred = torch.tensor(model.predict(zeo, osda).cpu().detach().numpy())
                syn_pred_list.append(syn_pred)
            syn_pred = torch.cat(syn_pred_list, dim=0).cpu().detach().numpy()
        
        elif configs['model_type'] == 'gan':
            assert cond_scale == None, 'cond_scale must not be provided for CVAE model'
            zeo_code = repeat(np.array(zeo_code), 'n -> (repeat n)', repeat=50)
            zeo = repeat(zeo, 'n d -> (repeat n) d', repeat=50)
            osda_smiles = repeat(np.array(osda_smiles), 'n -> (repeat n)', repeat=50)
            osda = repeat(osda, 'n d -> (repeat n) d', repeat=50)
            zeo, osda = zeo.to(configs['device']), osda.to(configs['device'])
            z = torch.randn(len(zeo), model.z_dims).to(configs['device'])
            syn_pred = torch.tensor(model(z, zeo, osda).cpu().detach().numpy())

        elif configs['model_type'] == 'nf':
            assert cond_scale == None, 'cond_scale must not be provided for CVAE model'
            zeo_code = repeat(np.array(zeo_code), 'n -> (repeat n)', repeat=50)
            zeo = repeat(zeo, 'n d -> (repeat n) d', repeat=50)
            osda_smiles = repeat(np.array(osda_smiles), 'n -> (repeat n)', repeat=50)
            osda = repeat(osda, 'n d -> (repeat n) d', repeat=50)
            zeo, osda = zeo.to(configs['device']), osda.to(configs['device'])
            z = torch.randn(len(zeo), model.syn_dims).to(configs['device'])
            syn_pred = torch.tensor(model.predict(z, zeo, osda).cpu().detach().numpy())

        elif configs['model_type'] == 'diff':
            print(f"Predicting using diffusion model with cond_scale of {cond_scale}")
            # n_samples = 50
            n_samples = 25
            zeo_code, osda_smiles = repeat(np.array(zeo_code), 'n -> (repeat n)', repeat=n_samples), repeat(np.array(osda_smiles), 'n -> (repeat n)', repeat=n_samples)
            zeo, osda = repeat(zeo, 'n d -> (repeat n) d', repeat=n_samples), repeat(osda, 'n d -> (repeat n) d', repeat=n_samples)
            zeo, osda = zeo.to(configs['device']), osda.to(configs['device'])
            chunks = []
            chunk_end_fracs = [0.2, 0.4, 0.6, 0.8, 1.0] # chunk_end_frac: fraction of sample at which chunk ends
            for chunk_end_frac in chunk_end_fracs: # Ensure all valid idxs
                assert (chunk_end_frac*zeo.shape[0]).is_integer(), 'each chunk must have whole number of samples'
            chunk_size = int(zeo.shape[0]/len(chunk_end_fracs)) # chunk_size: number of samples in each chunk
            for chunk_end_frac in chunk_end_fracs:
                chunk_end_idx = int(chunk_end_frac*zeo.shape[0]) # chunk_end_idx: index at which chunk ends
                chunk_start_idx = int(chunk_end_idx - chunk_size) # chunk_start_idx: index at which chunk starts   
                print(f'Predicting chunk from {chunk_start_idx} to {chunk_end_idx}...')
                chunk_sampled_data = model.sample(batch_size=chunk_size, zeo=zeo[chunk_start_idx:chunk_end_idx], osda=osda[chunk_start_idx:chunk_end_idx], cond_scale=cond_scale)
                chunk_syn_pred = torch.tensor(chunk_sampled_data.squeeze().detach().cpu().numpy())
                chunks.append(chunk_syn_pred)
            syn_pred = torch.cat(chunks, dim=0)

        elif configs['model_type'] == 'random':
            assert cond_scale == None, 'cond_scale must not be provided for CVAE model'
            zeo_code = repeat(np.array(zeo_code), 'n -> (repeat n)', repeat=50)
            zeo = repeat(zeo, 'n d -> (repeat n) d', repeat=50)
            osda_smiles = repeat(np.array(osda_smiles), 'n -> (repeat n)', repeat=50)
            osda = repeat(osda, 'n d -> (repeat n) d', repeat=50)

            # get all datapoints and get upper and lower limits
            syn = dataset.get_datapoints_by_index(range(len(dataset)), scaled=False, return_dataframe=True)[1]
            syn_pred = np.zeros([len(zeo), len(dataset.ratio_names)+len(dataset.cond_names)])
            for idx, col in enumerate(dataset.ratio_names+dataset.cond_names):
                syn_pred[:,idx] = np.random.uniform(syn.min(0)[col], syn.max(0)[col], len(zeo))

        elif configs['model_type'] == 'gmm':
            assert cond_scale == None, 'cond_scale must not be provided for CVAE model'
            zeo_code = repeat(np.array(zeo_code), 'n -> (repeat n)', repeat=50)
            zeo = repeat(zeo, 'n d -> (repeat n) d', repeat=50)
            osda_smiles = repeat(np.array(osda_smiles), 'n -> (repeat n)', repeat=50)
            osda = repeat(osda, 'n d -> (repeat n) d', repeat=50)

            X = torch.cat([zeo, osda], dim=1).numpy()

            syn_pred_chunks = []
            for i in range(0,len(X),2):
                print(i)
                try:
                    syn_pred = model.sample(X[i:i+2])[1]
                except Exception as e:
                    print(e)
                syn_pred_chunks.append(syn_pred)

            syn_pred = np.concatenate(syn_pred_chunks, axis=0)

        elif configs['model_type'] == 'bnn':
            assert cond_scale == None, 'cond_scale must not be provided for CVAE model'
            zeo_code = repeat(np.array(zeo_code), 'n -> (repeat n)', repeat=50)
            zeo = repeat(zeo, 'n d -> (repeat n) d', repeat=50)
            osda_smiles = repeat(np.array(osda_smiles), 'n -> (repeat n)', repeat=50)
            osda = repeat(osda, 'n d -> (repeat n) d', repeat=50)
            zeo_osda = torch.cat([zeo, osda], dim=1).to(configs['device'])
            syn_pred = torch.tensor(model(zeo_osda).cpu().detach().numpy())

        elif configs['model_type'] == 'nn':
            assert cond_scale == None, 'cond_scale must not be provided for CVAE model'
            zeo_code = repeat(np.array(zeo_code), 'n -> (repeat n)', repeat=50)
            zeo = repeat(zeo, 'n d -> (repeat n) d', repeat=50)
            osda_smiles = repeat(np.array(osda_smiles), 'n -> (repeat n)', repeat=50)
            osda = repeat(osda, 'n d -> (repeat n) d', repeat=50)
            zeo, osda = zeo.to(configs['device']), osda.to(configs['device'])
            syn_pred = torch.tensor(model(zeo, osda).cpu().detach().numpy())

        elif configs['model_type'] == 'amd':
            assert cond_scale == None, 'cond_scale must not be provided for CVAE model'
            # Zip and unzip to filter out 3 phases
            test_data = [(code, z, smiles, o) for (code, z, smiles, o) in zip(zeo_code, zeo, osda_smiles, osda) if code not in ['Dense/Amorphous', 'PTY', 'SFS']]
            zeo_code, zeo, osda_smiles, osda =  list(zip(*test_data))

            # Adhoc: 1) Get list of zeolite codes 2) Retrieval AMD features for zeolite from data/zeolite_amd_distance_matrix.csv
            amd_df = pd.read_csv('data/zeolite_amd_distance_matrix.csv').rename(columns={'Unnamed: 0': 'zeo'})
            test_zeo = [] # Note: test_zeo replaces zeo for AMD
            test_osda = []
            for z, o in zip(zeo_code, osda):
                test_zeo.append(torch.tensor(amd_df[amd_df['zeo']==z].drop(columns=['zeo']).values))
                test_osda.append(o.unsqueeze(0))
            zeo = torch.cat(test_zeo).float()
            osda = torch.cat(test_osda)

            # Rest same as nn
            zeo_code = repeat(np.array(zeo_code), 'n -> (repeat n)', repeat=50)
            zeo = repeat(zeo, 'n d -> (repeat n) d', repeat=50)
            osda_smiles = repeat(np.array(osda_smiles), 'n -> (repeat n)', repeat=50)
            osda = repeat(osda, 'n d -> (repeat n) d', repeat=50)
            zeo, osda = zeo.to(configs['device']), osda.to(configs['device'])
            syn_pred = torch.tensor(model(zeo, osda).cpu().detach().numpy())            

        # Scale synthesis conditions back
        if configs['model_type'] != 'random':
            for ratio_idx, ratio in enumerate(dataset.ratio_names+dataset.cond_names):
                qt = dataset.qts[ratio] # load quantile transformer
                syn_pred[:,ratio_idx] = torch.tensor(qt.inverse_transform(syn_pred[:,ratio_idx].reshape(-1, 1))).reshape(-1) # transform back

        # Save predictions
        syn_pred = pd.DataFrame(syn_pred, columns=dataset.ratio_names+dataset.cond_names)
        syn_pred['zeo'], syn_pred['osda'] = zeo_code, osda_smiles
        syn_pred.to_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/{pred_fname}", index=False)
        
    else:
        print(f"Loading synthetic predictions from saved predictions at runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/{pred_fname}")
        syn_pred = pd.read_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/{pred_fname}")

    syn_pred_scaled = utils.scale_x_syn_ratio(syn_pred, dataset) # get min-max scaled version too

    # Get ground truth
    syn_true, zeo_code, osda_smiles = _dataset[1], _dataset[3], _dataset[13]
    for ratio_idx, ratio in enumerate(dataset.ratio_names+dataset.cond_names):
        qt = dataset.qts[ratio] # load quantile transformer
        syn_true[:,ratio_idx] = torch.tensor(qt.inverse_transform(syn_true[:,ratio_idx].reshape(-1, 1))).reshape(-1) # transform back
    syn_true = pd.DataFrame(syn_true, columns=dataset.ratio_names+dataset.cond_names)
    syn_true['zeo'], syn_true['osda'] = zeo_code, osda_smiles

    # For AMD, filter out 3 phases that are not supported
    if configs['model_type'] == 'amd':
        syn_true = syn_true[(syn_true['zeo'] != 'Dense/Amorphous') & \
                            (syn_true['zeo'] != 'PTY') & \
                            (syn_true['zeo'] != 'SFS')
                            ]
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
    cov_zeo_agg = {} # Dict of coverages for each zeolite
    ae_zeo_agg = {} # Dict of AEs for each zeolite
    for zeo in zeo_systems:
        if zeo != 'Dense/Amorphous':
            if plot:
                print(zeo)

            sys_syn_pred, sys_syn_true = syn_pred[syn_pred['zeo'] == zeo], syn_true[syn_true['zeo'] == zeo]
            sys_syn_pred_scaled, sys_syn_true_scaled = syn_pred_scaled[syn_pred_scaled['zeo'] == zeo], syn_true_scaled[syn_true_scaled['zeo'] == zeo]
            
            if eval:
                # Coverage
                cov = coverage(sys_syn_pred[dataset.ratio_names+dataset.cond_names], sys_syn_true[dataset.ratio_names+dataset.cond_names])
                cov_zeo_agg[zeo] = cov

                # Regression metrics
                ae = abs_error(sys_syn_pred[dataset.ratio_names+dataset.cond_names], sys_syn_true[dataset.ratio_names+dataset.cond_names]) # use non-scaled version
                ae_zeo_agg[zeo] = ae

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
        # Save coverage metrics
        prec_zeo_agg_df, rec_zeo_agg_df = pd.DataFrame({'zeo': cov_zeo_agg.keys()}), pd.DataFrame({'zeo': cov_zeo_agg.keys()})

        for col in dataset.ratio_names+dataset.cond_names:
            prec_zeo_agg_df[col+'_prec'] = [x['precision'][col] for x in cov_zeo_agg.values()]
            rec_zeo_agg_df[col+'_rec'] = [x['recall'][col] for x in cov_zeo_agg.values()]
        
        prec_zeo_agg_df_mean, rec_zeo_agg_df_mean = prec_zeo_agg_df.mean(0), rec_zeo_agg_df.mean(0)

        f1_zeo_agg_df_mean = pd.DataFrame((prec_zeo_agg_df_mean.to_numpy()+rec_zeo_agg_df_mean.to_numpy())/2, index=dataset.ratio_names+dataset.cond_names)
        print(prec_zeo_agg_df_mean, rec_zeo_agg_df_mean, f1_zeo_agg_df_mean)

        if num_systems == None:
            prec_zeo_agg_df.to_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/prec_zeo_agg_df.csv", index=False)
            rec_zeo_agg_df.to_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/rec_zeo_agg_df.csv", index=False)

        # Save regression metrics
        ae_zeo_agg_df = pd.DataFrame({'zeo': ae_zeo_agg.keys()})
        regs = {}
        maes, wmaes, r2s = {}, {}, {}
        for col in dataset.ratio_names+dataset.cond_names:
            ae_zeo_agg_df[col+'_ae'] = [x[col]['ae'] for x in ae_zeo_agg.values()]
            ae_zeo_agg_df[col+'_pred_mean'] = [x[col]['pred_mean'] for x in ae_zeo_agg.values()]
            ae_zeo_agg_df[col+'_true_mean'] = [x[col]['true_mean'] for x in ae_zeo_agg.values()]

            # Mean absolute error (MAE)
            mae = ae_zeo_agg_df[col+'_ae'].mean()

            # Weighted mean absolute error (WMAE)
            weights_unnormed = (1./ae_zeo_agg_df[col+'_true_mean']).replace([np.inf, np.nan], 0)
            weights_normed = weights_unnormed/(weights_unnormed.sum())
            wmae = (weights_normed * ae_zeo_agg_df[col+'_ae']).sum() # weighted mean absolute error (weighted by inverse of ground truth)
            
            # R2
            ae_zeo_agg_df_ = ae_zeo_agg_df[[col+'_pred_mean', col+'_true_mean']].dropna()
            # r2 = r2_score(ae_zeo_agg_df_[col+'_pred_mean'], ae_zeo_agg_df_[col+'_true_mean'])
            r2 = None

            maes[col] = mae
            wmaes[col] = wmae
            r2s[col] = r2
        
        regs['maes'] = maes
        regs['wmaes'] = wmaes
        regs['r2s'] = r2s

        if num_systems == None:
            ae_zeo_agg_df.to_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/ae_zeo_agg_df.csv", index=False)
            with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/reg_zeo_agg.json", "w") as outfile:
                json.dump(regs, outfile, indent=4)

        # Save MMD
        mmd_zeo_agg_df = pd.DataFrame({'zeo': mmd_zeo_agg.keys(), 'MMD': mmd_zeo_agg.values()})
        assert mmd_zeo_agg_df['MMD'].isna().sum() == 0 # Check no NaNs
        mmd_mean, mmd_std = mmd_zeo_agg_df['MMD'].mean(), mmd_zeo_agg_df['MMD'].std()
        print('Mean MMD:', mmd_mean)
        if num_systems == None: # Save only if evaluated on all systems
            mmd_zeo_agg_df.to_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/mmd_zeo_agg_df.csv", index=False) # Save per-system MMDs
            with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/mmd_zeo_agg.json", "w") as outfile:
                json.dump({'MMD mean': mmd_mean, 'MMD std': mmd_std}, outfile, indent=4)

        # Save WSD
        wsd_zeo_agg_df = pd.DataFrame({'zeo': wsd_zeo_agg.keys(), 'WSD': wsd_zeo_agg.values()})
        assert wsd_zeo_agg_df['WSD'].isna().sum() == 0 # Check no NaNs
        wsd_mean, wsd_std = wsd_zeo_agg_df['WSD'].mean(), wsd_zeo_agg_df['WSD'].std()
        print('Mean WSD:', wsd_mean)
        if num_systems == None: # Save only if evaluated on all systems
            wsd_zeo_agg_df.to_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/wsd_zeo_agg_df.csv", index=False) # Save per-system WSDs
            with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/wsd_zeo_agg.json", "w") as outfile:
                json.dump({'WSD mean': wsd_mean, 'WSD std': wsd_std}, outfile, indent=4)
        
        return mmd_zeo_agg_df, wsd_zeo_agg_df
    
    else:
        pass

def eval_zeolite_osda(syn_pred, syn_pred_scaled, syn_true, syn_true_scaled, dataset, configs, eval=True, plot=False, print_metrics=False, num_systems=None, return_f1=False):
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
    cov_zeo_osda = {} # Dict of coverages for each zeolite
    ae_zeo_osda = {} # Dict of AEs for each zeolite-osda
    for zeo, osda in zeo_osda_systems:
        if zeo != 'Dense/Amorphous':
            if plot:
                print(zeo, osda)
            sys_syn_pred, sys_syn_true = syn_pred[(syn_pred['zeo'] == zeo) & (syn_pred['osda'] == osda)], syn_true[(syn_true['zeo'] == zeo) & (syn_true['osda'] == osda)]
            sys_syn_pred_scaled, sys_syn_true_scaled = syn_pred_scaled[(syn_pred_scaled['zeo'] == zeo) & (syn_pred_scaled['osda'] == osda)], syn_true_scaled[(syn_true_scaled['zeo'] == zeo) & (syn_true_scaled['osda'] == osda)]

            if eval:
                # Coverage
                cov = coverage(sys_syn_pred[dataset.ratio_names+dataset.cond_names], sys_syn_true[dataset.ratio_names+dataset.cond_names])
                cov_zeo_osda[(zeo,osda)] = cov

                # Regression metrics
                ae = abs_error(sys_syn_pred[dataset.ratio_names+dataset.cond_names], sys_syn_true[dataset.ratio_names+dataset.cond_names]) # use non-scaled version
                ae_zeo_osda[(zeo,osda)] = ae

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
                try:
                    utils.compare_gel_conds([sys_syn_pred, sys_syn_true], ['Predicted', 'True'], [True, False], [False, True], ['tab:orange', 'tab:blue'], common_norm=True, alpha=0.5)
                except Exception as e:
                    print('ERROR: ', e)

            count += 1
            
        if count == num_systems:
            break
    if eval:
        # Save coverage metrics
        prec_zeo_osda_df, rec_zeo_osda_df = pd.DataFrame({'zeo': [z for z, _ in [*cov_zeo_osda]], 'osda': [o for _, o in [*cov_zeo_osda]]}), pd.DataFrame({'zeo': [z for z, _ in [*cov_zeo_osda]], 'osda': [o for _, o in [*cov_zeo_osda]]})

        for col in dataset.ratio_names+dataset.cond_names:
            prec_zeo_osda_df[col+'_prec'] = [x['precision'][col] for x in cov_zeo_osda.values()]
            rec_zeo_osda_df[col+'_rec'] = [x['recall'][col] for x in cov_zeo_osda.values()]
        
        prec_zeo_osda_df_mean, rec_zeo_osda_df_mean = prec_zeo_osda_df.mean(0), rec_zeo_osda_df.mean(0)
        f1_zeo_osda_df_mean = pd.DataFrame((prec_zeo_osda_df_mean.to_numpy()+rec_zeo_osda_df_mean.to_numpy())/2, index=dataset.ratio_names+dataset.cond_names)
        print(prec_zeo_osda_df_mean, rec_zeo_osda_df_mean, f1_zeo_osda_df_mean)

        if num_systems == None:
            prec_zeo_osda_df.to_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/prec_zeo_osda_df.csv", index=False)
            rec_zeo_osda_df.to_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/rec_zeo_osda_df.csv", index=False)

        # Save regression metrics
        ae_zeo_osda_df = pd.DataFrame({'zeo': [z for z, _ in [*ae_zeo_osda]], 'osda': [o for _, o in [*ae_zeo_osda]]})
        regs = {}
        maes, wmaes, r2s = {}, {}, {}
        for col in dataset.ratio_names+dataset.cond_names:
            ae_zeo_osda_df[col+'_ae'] = [x[col]['ae'] for x in ae_zeo_osda.values()]
            ae_zeo_osda_df[col+'_pred_mean'] = [x[col]['pred_mean'] for x in ae_zeo_osda.values()]
            ae_zeo_osda_df[col+'_true_mean'] = [x[col]['true_mean'] for x in ae_zeo_osda.values()]

            # Mean absolute error (MAE)
            mae = ae_zeo_osda_df[col+'_ae'].mean()

            # Weighted mean absolute error (WMAE)
            weights_unnormed = (1./ae_zeo_osda_df[col+'_true_mean']).replace([np.inf, np.nan], 0)
            weights_normed = weights_unnormed/(weights_unnormed.sum())
            wmae = (weights_normed * ae_zeo_osda_df[col+'_ae']).sum() # weighted mean absolute error (weighted by inverse of ground truth)
            
            # R2
            ae_zeo_osda_df_ = ae_zeo_osda_df[[col+'_pred_mean', col+'_true_mean']].dropna()
            # r2 = r2_score(ae_zeo_osda_df_[col+'_pred_mean'], ae_zeo_osda_df_[col+'_true_mean'])
            r2 = None

            maes[col] = mae
            wmaes[col] = wmae
            r2s[col] = r2
        
        regs['maes'] = maes
        regs['wmaes'] = wmaes
        regs['r2s'] = r2s

        if num_systems == None:
            ae_zeo_osda_df.to_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/ae_zeo_osda_df.csv", index=False)
            with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/reg_zeo_osda.json", "w") as outfile:
                json.dump(regs, outfile, indent=4)

        # Save MMD
        mmd_zeo_osda_df = pd.DataFrame({'zeo': [z for z, _ in [*mmd_zeo_osda]], 'osda': [o for _, o in [*mmd_zeo_osda]], 'MMD': mmd_zeo_osda.values()})
        assert mmd_zeo_osda_df['MMD'].isna().sum() == 0 # Check no NaNs
        mmd_mean, mmd_std = mmd_zeo_osda_df['MMD'].mean(), mmd_zeo_osda_df['MMD'].std()
        print('Mean MMD:', mmd_mean)
        if num_systems == None: # Save only if evaluated on all systems
            mmd_zeo_osda_df.to_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/mmd_zeo_osda_df.csv", index=False) # Save per-system MMDs
            with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/mmd_zeo_osda.json", "w") as outfile:
                json.dump({'MMD mean': mmd_mean, 'MMD std': mmd_std}, outfile, indent=4)

        # Save WSD
        wsd_zeo_osda_df = pd.DataFrame({'zeo': [z for z, _ in [*wsd_zeo_osda]], 'osda': [o for _, o in [*wsd_zeo_osda]], 'WSD': wsd_zeo_osda.values()})
        assert wsd_zeo_osda_df['WSD'].isna().sum() == 0 # Check no NaNs
        wsd_mean, wsd_std = wsd_zeo_osda_df['WSD'].mean(), wsd_zeo_osda_df['WSD'].std()
        print('Mean WSD:', wsd_mean)
        if num_systems == None: # Save only if evaluated on all systems
            wsd_zeo_osda_df.to_csv(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/wsd_zeo_osda_df.csv", index=False) # Save per-system WSDs
            with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/wsd_zeo_osda.json", "w") as outfile:
                json.dump({'WSD mean': wsd_mean, 'WSD std': wsd_std}, outfile, indent=4)

        if return_f1:
            return mmd_zeo_osda_df, wsd_zeo_osda_df, prec_zeo_osda_df_mean, rec_zeo_osda_df_mean, f1_zeo_osda_df_mean
        else:
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

def plot_single_system(syn_pred, syn_true, zeo, osda=None):
    if osda == None: # Zeolite-aggregated
        print(zeo)
        sys_syn_pred, sys_syn_true = syn_pred[syn_pred['zeo'] == zeo], syn_true[syn_true['zeo'] == zeo]

    else: # Zeolite-OSDA
        print(zeo, osda)
        sys_syn_pred, sys_syn_true = syn_pred[(syn_pred['zeo'] == zeo) & (syn_pred['osda'] == osda)], syn_true[(syn_true['zeo'] == zeo) & (syn_true['osda'] == osda)]
        
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
    ### Single model evaluation ####
    for model_type, fname, split in [
                                    # ('random', 'v0', 'system'),
                                    # ('amd', 'v0', 'system'),
                                    # ('nn', 'v0', 'system'),
                                    # ('bnn', 'v2', 'system'),
                                    # ('gmm', 'v0', 'system'),
                                    # ('gan', 'v3-3', 'system'),
                                    # ('nf', 'v0', 'system'),
                                    # ('cvae', 'v10', 'system'),
                                    # ('cvae-eq', 'v4', 'system'),
                                    # ('cvae-gnn', 'v3', 'system'),
                                    ('diff', 'run1', 'system'),
                                    ]:
    # # for model_type, fname, split in [
    #                                 # ('amd', 'v0-2', 'system'),
    #                                 # ('amd', 'v0-3', 'system'),
    #                                 # ('bnn', 'v3', 'system'),
    #                                 # ('gmm', 'v0-2', 'system'),
    #                                 # ('gmm', 'v0-3', 'system'),
    #                                 # ('nf', 'v0-2', 'system'),
    #                                 # ('nf', 'v0-3', 'system'),
    #                                 # ('cvae', 'v10-2', 'system'),
    #                                 # ('cvae', 'v10-3', 'system'),
    #                                 # ]:
    #     model, configs = load_model(model_type, fname, split)
    #     syn_pred, syn_pred_scaled, syn_true, syn_true_scaled, dataset = get_prediction_and_ground_truths(model, configs)
    #     # mmd_zeo_agg_df, wsd_zeo_agg_df = eval_zeolite_aggregated(syn_pred, syn_pred_scaled, syn_true, syn_true_scaled, dataset, configs)
    #     mmd_zeo_osda_df, wsd_zeo_osda_df = eval_zeolite_osda(syn_pred, syn_pred_scaled, syn_true, syn_true_scaled, dataset, configs)

        #### Single diffusion model evaluation + Vary cond_scale ####
        for cond_scale in [0.75]:
            model, configs = load_model(model_type, fname, split)
            syn_pred, syn_pred_scaled, syn_true, syn_true_scaled, dataset = get_prediction_and_ground_truths(model, configs, cond_scale=cond_scale)
            mmd_zeo_osda_df, wsd_zeo_osda_df = eval_zeolite_osda(syn_pred, syn_pred_scaled, syn_true, syn_true_scaled, dataset, configs)

    # #### Multiple diffusion model evaluation + Vary cond_scales ####
    # model_type = 'diff'
    # split = 'system'
    # for fname in [
    #               'run1',
    #               ]:
    #     for cond_scale in [0.75]:
    #         model, configs = load_model(model_type, fname, split)
    #         syn_pred, syn_pred_scaled, syn_true, syn_true_scaled, dataset = get_prediction_and_ground_truths(model, configs, cond_scale=cond_scale)
    #         mmd_zeo_osda_df, wsd_zeo_osda_df = eval_zeolite_osda(syn_pred, syn_pred_scaled, syn_true, syn_true_scaled, dataset, configs)
