import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, generator_layer_size, z_dims, zeo_h_dims=64, osda_h_dims=64, syn_dims=12, zeo_feat_dims=143, osda_feat_dims=14):
        super().__init__()
        
        self.z_dims = z_dims
        self.zeo_h_dims = zeo_h_dims
        self.osda_h_dims = osda_h_dims
        self.syn_dims = syn_dims
        self.zeo_feat_dims  = zeo_feat_dims
        self.osda_feat_dims = osda_feat_dims

        self.zeo_mlp = nn.Sequential(nn.Linear(self.zeo_feat_dims, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, self.zeo_h_dims),
                                    )
        
        self.osda_mlp = nn.Sequential(nn.Linear(self.osda_feat_dims, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, self.osda_h_dims),
                                    )

        self.model = nn.Sequential(
            nn.Linear(self.z_dims+self.zeo_h_dims+self.osda_h_dims, generator_layer_size[0]),
            nn.ReLU(),
            nn.Linear(generator_layer_size[0], generator_layer_size[1]),
            nn.ReLU(),
            nn.Linear(generator_layer_size[1], generator_layer_size[2]),
            nn.ReLU(),
            nn.Linear(generator_layer_size[2], syn_dims),
            nn.Sigmoid()
        )

    
    def forward(self, z, zeo, osda):
        
        if len(z) != len(zeo): # Ad-hoc fix for last mini-batch of each epoch is != BATCH_SIZE
            z = z[:len(zeo)]
        
        # Encode zeolite and osda
        zeo = self.zeo_mlp(zeo)
        osda = self.osda_mlp(osda)

        # Concat z & label
        x = torch.cat([z, zeo, osda], dim = -1)
        
        return self.model(x)
    

class Discriminator(nn.Module):
    def __init__(self, discriminator_layer_size, zeo_h_dims=64, osda_h_dims=64, syn_dims=12, zeo_feat_dims=143, osda_feat_dims=14):
        super().__init__()
        
        self.zeo_h_dims = zeo_h_dims
        self.osda_h_dims = osda_h_dims
        self.syn_dims = syn_dims
        self.zeo_feat_dims  = zeo_feat_dims
        self.osda_feat_dims = osda_feat_dims

        self.zeo_mlp = nn.Sequential(nn.Linear(self.zeo_feat_dims, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, self.zeo_h_dims),
                                    )
        
        self.osda_mlp = nn.Sequential(nn.Linear(self.osda_feat_dims, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, self.osda_h_dims),
                                    )
        
        self.model = nn.Sequential(
            nn.Linear(self.syn_dims+ self.zeo_h_dims+self.osda_h_dims, discriminator_layer_size[0]),
            nn.ReLU(),
            nn.Linear(discriminator_layer_size[0], discriminator_layer_size[1]),
            nn.ReLU(),
            nn.Linear(discriminator_layer_size[1], discriminator_layer_size[2]),
            nn.ReLU(),
            nn.Linear(discriminator_layer_size[2], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, zeo, osda):

        # Encode zeolite and osda
        zeo = self.zeo_mlp(zeo)
        osda = self.osda_mlp(osda)
        
        # Concat image & label
        x = torch.cat([x, zeo, osda], dim = -1)
        
        return self.model(x)