import os
import sys
import json
from models.diffusion import *
import data.utils as utils
sys.modules['utils'] = utils # Way to get around relative imports in utils for ZeoSynGen_dataset # https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
import pdb


configs = {
            "model_type" : "diff",
            "split" : "system",
            "fname": "test",
            "device" : "cuda:0",
            "train_batch_size": 128, # optimal 128
            "train_lr": 4e-4, # optimal 4e-4
            "train_num_steps": 1e6, # optimal 1e6
            "gradient_accumulate_every": 2, # optimal 2
            "ema_decay": 0.9, # optimal 0.9
            "amp": False,
            "lr_decay": False,
            "lr_decay_gamma": 0.99999,
            "save_all_model_checkpoints": True,
            "model_params":{
                            "dim": 12,
                            "dim_mults": (64, 128, 256), # optimal (64, 128, 256)
                            "channels": 1,
                            "resnet_block_groups": 4,
                            "cond_drop_prob": 0.10, # optimal 0.10
                            "seq_length": 12,
                            "timesteps": 100, # optimal 100
                            "zeo_feat_dims": 143, 
                            "osda_feat_dims": 14,
                            "zeo_h_dims": 64, 
                            "osda_h_dims": 64,
                            "dropout": False,
                            },
            }

def train_diff(configs):

    # Restrict visible devices to avoid cross-GPU memory leakage (due to accelerate) hence using .to('cuda')
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=configs['device'].split(":")[-1]

    # Create run folder
    assert os.path.isdir(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}") == False, 'Name already taken. Please choose another folder name.'
    os.mkdir(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}")

    # Save configs
    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/configs.json", "w") as outfile:
        json.dump(configs, outfile, indent=4)

    # 1) Initialize model and trainer
    model = Unet1D(
            dim                 = configs['model_params']['dim'],
            dim_mults           = configs['model_params']['dim_mults'],
            channels            = configs['model_params']['channels'],
            resnet_block_groups = configs['model_params']['resnet_block_groups'],
            zeo_feat_dims       = configs['model_params']['zeo_feat_dims'],
            osda_feat_dims      = configs['model_params']['osda_feat_dims'],
            cond_drop_prob      = configs['model_params']['cond_drop_prob'],
            dropout             = configs['model_params']['dropout'],
            )

    diffusion = GaussianDiffusion1D(
            model,
            seq_length = configs['model_params']['seq_length'],
            timesteps  = configs['model_params']['timesteps'],
            objective  = 'pred_v',
            ).to('cuda')

    
    # 2) Load dataset
    with open(f'data/ZeoSynGen_dataset.pkl', 'rb') as f: # load dataset
        dataset = pickle.load(f)
    train_dataset, val_dataset, test_dataset = dataset.train_val_test_split(mode=configs['split'], both_graph_feat_present=True, random_state=0) # Note, here we filter out points with no graph/feature present for either zeolite and OSDA
    syn_train, zeo_train, osda_train = train_dataset[1], train_dataset[5], train_dataset[15]
    syn_val, zeo_val, osda_val = val_dataset[1], val_dataset[5], val_dataset[15]

    trainer = Trainer1D(
            diffusion,
            syn_train                 = syn_train,
            zeo_train                 = zeo_train,
            osda_train                = osda_train,
            syn_val                   = syn_val,
            zeo_val                   = zeo_val,
            osda_val                  = osda_val,
            train_batch_size          = configs['train_batch_size'],
            train_lr                  = configs['train_lr'],
            train_num_steps           = configs['train_num_steps'], # total training steps
            gradient_accumulate_every = configs['gradient_accumulate_every'],     # gradient accumulation steps
            ema_decay                 = configs['ema_decay'], # exponential moving average decay
            amp                       = configs['amp'],  # turn on mixed precision
            lr_decay                  = configs['lr_decay'],
            lr_decay_gamma            = configs['lr_decay_gamma'],
            model_save_path           = f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}",
            save_all_model_checkpoints = configs['save_all_model_checkpoints'],
            device                    = 'cuda' # 'cuda' used instead of configs['device'] to avoid cross-GPU memory leakage
            )
    
    diffusion, train_loss_list, val_loss_list = trainer.train()

    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/train_loss_list.pkl", 'wb') as handle:
        pickle.dump(train_loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/val_loss_list.pkl", 'wb') as handle:
        pickle.dump(val_loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    train_diff(configs)
