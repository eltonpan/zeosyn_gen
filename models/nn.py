import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, zeo_h_dims=64, osda_h_dims=64, syn_dims=12, zeo_feat_dims=143, osda_feat_dims=14):
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
        
        self.fusion_mlp = nn.Sequential(nn.Linear(self.zeo_h_dims+self.osda_h_dims, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, self.syn_dims),
                                     nn.Sigmoid(),
                                    )
        
    def forward(self, zeo, osda):

        zeo = self.zeo_mlp(zeo)
        osda = self.osda_mlp(osda)
        h = torch.cat([zeo, osda], dim = -1)

        return self.fusion_mlp(h)