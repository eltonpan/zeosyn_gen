import torch
import torch.nn as nn

class ConditionalAffineCoupling(nn.Module):
    def __init__(self, hidden_size, zeo_h_dims=64, osda_h_dims=64, syn_dims=12, zeo_feat_dims=143, osda_feat_dims=14):
        super(ConditionalAffineCoupling, self).__init__()
        
        self.hidden_size = hidden_size
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

        self.net = nn.Sequential(
            nn.Linear(int(self.syn_dims*0.5)+self.zeo_h_dims+self.osda_h_dims, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, syn_dims),
            nn.Tanh()
        )
        
        
    def forward(self, x, zeo, osda, layer_idx):

        # Encode zeolite and osda
        zeo = self.zeo_mlp(zeo)
        osda = self.osda_mlp(osda)
        
        x0, x1 = x.chunk(2, dim=1)
        
        if layer_idx%2 == 0: 
            log_s, t = self.net(torch.cat([x0, zeo, osda], dim=1)).chunk(2, dim=1)
            s = torch.exp(log_s)
            y1 = s * x1 + t
            y0 = x0
            y = torch.cat([y0, y1], dim=1)
            log_det = torch.sum(torch.log(s), dim=1)
        else:
            log_s, t = self.net(torch.cat([x1, zeo, osda], dim=1)).chunk(2, dim=1)
            s = torch.exp(log_s)
            y0 = s * x0 + t
            y1 = x1
            y = torch.cat([y0, y1], dim=1)
            log_det = torch.sum(torch.log(s), dim=1)
    
        return y, log_det
    
    def backward(self, y, zeo, osda, layer_idx):
        
        y0, y1 = y.chunk(2, dim=1)

        # Encode zeolite and osda
        zeo = self.zeo_mlp(zeo)
        osda = self.osda_mlp(osda)
        
        if layer_idx%2 == 0:
            log_s, t = self.net(torch.cat([y0, zeo, osda], dim=1)).chunk(2, dim=1)
            s = torch.exp(log_s) 
            x1 = (y1 - t) / s
            x0 = y0
            x = torch.cat([x0, x1], dim=1)
            log_det = torch.sum(torch.log(s), dim=1)
        else:
            log_s, t = self.net(torch.cat([y1, zeo, osda], dim=1)).chunk(2, dim=1)
            s = torch.exp(log_s) 
            x0 = (y0 - t) / s
            x1 = y1
            x = torch.cat([x0, x1], dim=1)
            log_det = torch.sum(torch.log(s), dim=1)
            
        return x, log_det
        

class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, num_flows, hidden_size, zeo_h_dims=64, osda_h_dims=64, syn_dims=12, zeo_feat_dims=143, osda_feat_dims=14):
        super(ConditionalNormalizingFlow, self).__init__()

        self.num_flows = num_flows
        self.hidden_size = hidden_size

        self.zeo_h_dims = zeo_h_dims
        self.osda_h_dims = osda_h_dims
        self.syn_dims = syn_dims
        self.zeo_feat_dims  = zeo_feat_dims
        self.osda_feat_dims = osda_feat_dims
        
        self.transforms = nn.ModuleList([
            ConditionalAffineCoupling(hidden_size=self.hidden_size, zeo_h_dims=self.zeo_h_dims, osda_h_dims=self.osda_h_dims, syn_dims=self.syn_dims, zeo_feat_dims=self.zeo_feat_dims, osda_feat_dims=self.osda_feat_dims)
            for _ in range(num_flows)
        ])
        self.layer_idxs = [idx for idx in range(num_flows)]
        
    def forward(self, x, zeo, osda):
        log_det = torch.zeros_like(x[:, 0])
        
        for transform, layer_idx in zip(self.transforms, self.layer_idxs):
            x, ld = transform(x, zeo, osda, layer_idx)
            log_det += ld
        
        return x, log_det
    
    def predict(self, z, zeo, osda):

        # Reverse the flow and transform the sample back to the original space
        with torch.no_grad():
            for transform, layer_idx in zip(reversed(self.transforms), reversed(self.layer_idxs)):
                z, _ = transform.backward(z, zeo, osda, layer_idx)
                
#         z = torch.sigmoid(z) # constrain output to 0-1
        
        return z