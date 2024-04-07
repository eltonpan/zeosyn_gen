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
from models.gan import Generator, Discriminator
from torch.autograd import Variable

configs = { 
            'model_type' : 'gan',
            'split' : 'system',
            'fname': 'v0',
            'device' : 'cuda:0',
            'batch_size' : 2048,
            'n_epochs' : 100, # 2e4
            'lr' : 1e-4,
            'model_params':{
                        'z_dims': 10,
                        'generator_layer_size': [16, 32, 64],
                        'discriminator_layer_size' :[64, 32, 16],
                        'zeo_h_dims': 64, 
                        'osda_h_dims': 64, 
                        'syn_dims': 12, 
                        'zeo_feat_dims': 143, 
                        'osda_feat_dims': 14,
                        },
            }


def generator_step(batch_size, discriminator, generator, g_optimizer, criterion, zeo, osda, train=True):
    
    if train:
        # Init gradient
        g_optimizer.zero_grad()
    
    # Building z
    z = Variable(torch.randn(batch_size, configs['model_params']['z_dims'])).to(configs['device'])
    
    # Building fake conditions
    # fake_cond = cond
    fake_zeo = zeo
    fake_osda = osda
    
    # Generating fake data
    fake_x = generator(z, fake_zeo, fake_osda)
    
    # Disciminating fake data
    validity = discriminator(fake_x, fake_zeo, fake_osda)
    
    # Calculating discrimination loss (fake images)
    g_loss = criterion(validity.squeeze(), Variable(torch.ones_like(validity.squeeze())).to(configs['device']))
    
    if train:
        # Backword propagation
        g_loss.backward()
        
        #  Optimizing generator
        g_optimizer.step()
    
    return g_loss.data

def discriminator_step(batch_size, discriminator, generator, d_optimizer, criterion, real_x, zeo, osda, train=True):
    
    if train:
        # Init gradient 
        d_optimizer.zero_grad()

    # Disciminating real images
    real_validity = discriminator(real_x, zeo, osda)
    
    # Calculating discrimination loss (real x)
    real_loss = criterion(real_validity.squeeze(), Variable(torch.ones(batch_size)).to(configs['device']))
    
    # Building z
    z = Variable(torch.randn(batch_size, configs['model_params']['z_dims'])).to(configs['device'])
    
    # Building fake labels
    # fake_cond = cond.to(configs['device'])
    fake_zeo = zeo.to(configs['device'])
    fake_osda = osda.to(configs['device'])
    
    # Generating fake images
    # fake_x = generator(z, fake_cond)
    fake_x = generator(z, fake_zeo, fake_osda)
    
    # Disciminating fake images
    # fake_validity = discriminator(fake_x, fake_cond)
    fake_validity = discriminator(fake_x, fake_zeo, fake_osda)
    
    # Calculating discrimination loss (fake images)
    fake_loss = criterion(fake_validity.squeeze(), Variable(torch.zeros(batch_size)).to(configs['device']))
    
    # Sum two losses
    d_loss = real_loss + fake_loss
    
    if train:
        # Backword propagation
        d_loss.backward()
    
        # Optimizing discriminator
        d_optimizer.step()
    
    return d_loss.data

def train_gan(generator, discriminator, configs):

    # Create run folder
    assert os.path.isdir(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}") == False, 'Name already taken. Please choose another folder name.'
    os.mkdir(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}")

    # Save configs
    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/configs.json", "w") as outfile:
        json.dump(configs, outfile, indent=4)

    generator.to(configs['device'])
    discriminator.to(configs['device'])

    with open(f'data/ZeoSynGen_dataset.pkl', 'rb') as f: # load dataset
        dataset = pickle.load(f)

    train_dataset, val_dataset, _ = dataset.train_val_test_split(mode=configs['split'], both_graph_feat_present=True, random_state=0) # Note, here we filter out points with no graph/feature present for either zeolite and OSDA

    train_dataset = (train_dataset[1], train_dataset[5], train_dataset[15])
    val_dataset   = (val_dataset[1],   val_dataset[5],   val_dataset[15])

    train_loader = DataLoader(list(zip(*train_dataset)), batch_size=configs['batch_size'], shuffle=True)
    val_loader  = DataLoader(list(zip(*val_dataset)),  batch_size=configs['batch_size'], shuffle=False)


    # Loss function
    criterion = nn.BCELoss()

    # Optimizer
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=configs['lr'])
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=configs['lr'])

    g_best_model = None # Model to be outputted based on val loss
    d_best_model = None # Model to be outputted based on val loss
    g_train_loss_list  = []
    g_val_loss_list    = []
    d_train_loss_list  = []
    d_val_loss_list    = []

    best_val_loss = 1e10


    for epoch in tqdm(range(configs['n_epochs'])):

        for (syn, zeo, osda) in train_loader:
            # Set generator train
            generator.train()
            discriminator.train()

            # Train data
            real_syn = Variable(syn).to(configs['device'])
            # cond = Variable(torch.cat([zeo, osda, comp], dim = 1)).to(configs['device'])
            zeo = Variable(zeo).to(configs['device'])
            osda = Variable(osda).to(configs['device'])

            # Train discriminator
            d_loss_train = discriminator_step(len(real_syn), discriminator,
                                              generator, d_optimizer, criterion,
                                              real_syn, zeo, osda)

            # Train generator
            g_loss_train = generator_step(configs['batch_size'], discriminator, generator, g_optimizer, criterion, zeo, osda)

        if epoch%10 == 0:
            for (syn, zeo, osda) in val_loader:
            # Set generator eval
                generator.eval()
                discriminator.eval()

                # Train data
                real_syn = Variable(syn).to(configs['device'])
                # cond = Variable(torch.cat([zeo, osda, comp], dim = 1)).to(configs['device'])
                zeo = Variable(zeo).to(configs['device'])
                osda = Variable(osda).to(configs['device'])

                # Train discriminator
                d_loss_val = discriminator_step(len(real_syn), discriminator,
                                                generator, d_optimizer, criterion,
                                                real_syn, zeo, osda, train=False)

                # Train generator
                g_loss_val = generator_step(configs['batch_size'], discriminator, generator, g_optimizer, criterion, zeo, osda, train=False)

                tot_loss_val = d_loss_val+g_loss_val
                # Save best model according to minima of valid loss
                if g_loss_val < best_val_loss: # if val loss has decreased
                    best_val_loss = g_loss_val # update best val loss
                    best_gen, best_disc = generator, discriminator # update model
                    print()
                    print('Best model updated at Epoch {}'.format(epoch))
                    torch.save(generator.state_dict(), f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/gen.pt")
                    torch.save(discriminator.state_dict(), f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/disc.pt")
                    print()

                print('VAL Epoch {} g_loss: {}, d_loss: {}'.format(epoch, g_loss_val, d_loss_val))

        g_train_loss_list.append(g_loss_train.item())
        d_train_loss_list.append(d_loss_train.item())
        g_val_loss_list.append(g_loss_val.item())
        d_val_loss_list.append(d_loss_val.item())

        print('TRAIN Epoch {} g_loss: {}, d_loss: {}'.format(epoch, g_loss_train, d_loss_train))
    
    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/g_train_loss_list.pkl", 'wb') as file:
        pickle.dump(g_train_loss_list, file)
    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/g_val_loss_list.pkl", 'wb') as file:
        pickle.dump(g_val_loss_list, file)
    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/d_train_loss_list.pkl", 'wb') as file:
        pickle.dump(d_train_loss_list, file)
    with open(f"runs/{configs['model_type']}/{configs['split']}/{configs['fname']}/d_val_loss_list.pkl", 'wb') as file:
        pickle.dump(d_val_loss_list, file)

if __name__ == '__main__':
    generator = Generator(generator_layer_size=configs['model_params']['generator_layer_size'], z_dims=configs['model_params']['z_dims'])
    discriminator = Discriminator(discriminator_layer_size=configs['model_params']['discriminator_layer_size'])
    train_gan(generator, discriminator, configs)