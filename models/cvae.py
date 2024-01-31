import torch
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, z_dims=64, syn_dims=12, zeo_feat_dims=143, osda_feat_dims=14):
        super().__init__()
        self.z_dims   = z_dims
        self.syn_dims = syn_dims
        self.zeo_feat_dims  = zeo_feat_dims
        self.osda_feat_dims = osda_feat_dims


        self.encoder = nn.Sequential(nn.Linear(self.syn_dims + self.zeo_feat_dims + self.osda_feat_dims, 4096),
                                     nn.ReLU(),
                                     nn.Linear(4096, 2048),
                                     nn.ReLU(),
                                     nn.Linear(2048, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 512),
                                     nn.ReLU(),
                                    )
        
        self.mu      = nn.Linear(512, self.z_dims)
        self.logvar  = nn.Linear(512, self.z_dims)

        self.decoder = nn.Sequential(nn.Linear(self.z_dims + self.zeo_feat_dims + self.osda_feat_dims, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 2048),
                                     nn.ReLU(),
                                     nn.Linear(2048, 4096),
                                     nn.ReLU(),
                                     nn.Linear(4096, self.syn_dims),
                                     nn.Sigmoid()
        )

    def forward(self, x, zeo, osda):
        # CVAE forward function
        x = torch.cat([x, zeo, osda], dim = -1)
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        var = torch.exp(logvar) # exponentiate to get variance from log-variance
        std = torch.sqrt(var)
        
        # sample z - Sampling from N(0, 1) scaling it by the std, and adding mu is equivalent to sampling from N(mu, std^2)
        z = torch.randn_like(mu) # sample from N(0, 1) 
        z *= std # scale z by the std to give N(0, std^2)
        z += mu # add mu to give N(mu, std)
        
        # condition by concantenation
        z = torch.cat([z, zeo, osda], dim = -1)

        x_prime = self.decoder(z) # reconstructed x
        
        return x_prime, mu, logvar
    
    def predict(self, zeo, osda):

        # Sample from the prior distribution
        z = torch.randn(zeo.shape[0], self.z_dims).to(zeo.device)
        z = torch.cat([z, zeo, osda], dim = -1)
        return self.decoder(z)