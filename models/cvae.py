import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

class CVAEv1(nn.Module):
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
        z = torch.randn(zeo.shape[0], self.z_dims).to(f'cuda:{zeo.get_device()}')
        z = torch.cat([z, zeo, osda], dim = -1)
        return self.decoder(z)

class CVAEv2(nn.Module):
    def __init__(self, z_dims=64, zeo_h_dims=64, osda_h_dims=64, syn_dims=12, zeo_feat_dims=143, osda_feat_dims=14):
        super().__init__()
        self.z_dims   = z_dims
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
        
        self.encoder = nn.Sequential(nn.Linear(self.syn_dims+self.zeo_h_dims+self.osda_h_dims, 4096),
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

        self.decoder = nn.Sequential(nn.Linear(self.z_dims+self.zeo_h_dims+self.osda_h_dims, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 2048),
                                     nn.ReLU(),
                                     nn.Linear(2048, 4096),
                                     nn.ReLU(),
                                     nn.Linear(4096, self.syn_dims),
                                     nn.Sigmoid(),
                                    )

    def forward(self, x, zeo, osda):
        
        # Encode zeolite and osda
        zeo = self.zeo_mlp(zeo)
        osda = self.osda_mlp(osda)

        # Concatenate zeolite and osda to the input
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

        # Encode zeolite and osda
        zeo = self.zeo_mlp(zeo)
        osda = self.osda_mlp(osda)

        # Sample from the prior distribution
        z = torch.randn(zeo.shape[0], self.z_dims).to(f'cuda:{zeo.get_device()}')
        z = torch.cat([z, zeo, osda], dim = -1)
        return self.decoder(z)
    
class EGNN(torch.nn.Module):
    def __init__(self, lmax=3, out_feat=64, n_layers=1, num_neighbors=2, num_nodes=4, mul=5) -> None:
        self.lmax = lmax
        self.out_feat = out_feat
        self.n_layers = n_layers
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.mul = mul

        super().__init__()
        self.irreps_sh = o3.Irreps.spherical_harmonics(self.lmax)

        irreps_mid = o3.Irreps([(self.mul, (l, p)) for l in range(self.lmax + 1) for p in [-1, 1]])
        irreps_out = o3.Irreps(f"{self.out_feat}x0e")

        self.tp_in = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_sh,
            irreps_in2=self.irreps_sh,
            irreps_out=irreps_mid,
        )

        self.tp_layers = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.tp_layers.append(
                FullyConnectedTensorProduct(
                    irreps_in1=irreps_mid,
                    irreps_in2=self.irreps_sh,
                    irreps_out=irreps_mid,
                )
            )

        self.tp_out = FullyConnectedTensorProduct(
            irreps_in1=irreps_mid,
            irreps_in2=self.irreps_sh,
            irreps_out=irreps_out,
        )
        self.irreps_out = self.tp_out.irreps_out

    def forward(self, data) -> torch.Tensor:
        edge_src, edge_dst = (
            data.edge_index[0],
            data.edge_index[1],
        )  # tensors of indices representing the graph
        edge_vec = data.edge_vec

        edge_sh = o3.spherical_harmonics(
            l=self.irreps_sh,
            x=edge_vec,
            normalize=False,  # here we don't normalize otherwise it would not be a polynomial
            normalization="component",
        )

        node_features = scatter(edge_sh, edge_dst, dim=0).div(self.num_neighbors**0.5)

        edge_features = self.tp_in(node_features[edge_src], edge_sh)
        node_features = scatter(edge_features, edge_dst, dim=0).div(self.num_neighbors**0.5)

        for tp_layers in self.tp_layers:
            edge_features = tp_layers(node_features[edge_src], edge_sh)
            node_features = scatter(edge_features, edge_dst, dim=0).div(self.num_neighbors**0.5)

        edge_features = self.tp_out(node_features[edge_src], edge_sh)
        node_features = scatter(edge_features, edge_dst, dim=0).div(self.num_neighbors**0.5)

        # For each graph, all the node's features are summed
        output = scatter(node_features, data.batch, dim=0).div(self.num_nodes**0.5)/10000

        return output
    
class CVAE_EQ(nn.Module):
    def __init__(self, z_dims=64, zeo_h_dims=64, osda_feat_dims=14, osda_h_dims=64, syn_dims=12, lmax=2, n_layers=1, num_neighbors=9.714, num_nodes=138.2):
        super().__init__()
        self.z_dims   = z_dims
        self.zeo_h_dims = zeo_h_dims
        self.osda_h_dims = osda_h_dims
        self.syn_dims = syn_dims
        self.osda_feat_dims = osda_feat_dims
        self.lmax = lmax
        self.n_layers = n_layers
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        
        self.zeo_egnn = EGNN(lmax=self.lmax, out_feat=self.zeo_h_dims, n_layers=self.n_layers, num_neighbors=self.num_neighbors, num_nodes=self.num_nodes)
        self.zeo_mlp = self.osda_mlp = nn.Sequential(nn.Linear(self.zeo_h_dims, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, zeo_h_dims),
                                    #  nn.Tanh(),
                                    )
        self.osda_mlp = nn.Sequential(nn.Linear(self.osda_feat_dims, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, self.osda_h_dims),
                                    )
        
        self.encoder = nn.Sequential(nn.Linear(self.syn_dims+self.zeo_h_dims+self.osda_h_dims, 4096),
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

        self.decoder = nn.Sequential(nn.Linear(self.z_dims+self.zeo_h_dims+self.osda_h_dims, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 2048),
                                     nn.ReLU(),
                                     nn.Linear(2048, 4096),
                                     nn.ReLU(),
                                     nn.Linear(4096, self.syn_dims),
                                     nn.Sigmoid(),
                                    )

    def forward(self, x, zeo, osda):
        
        # Encode zeolite and osda
        zeo = self.zeo_mlp(self.zeo_egnn(zeo))
        osda = self.osda_mlp(osda)

        # Concatenate zeolite and osda to the input
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

        # Encode zeolite and osda
        zeo = self.zeo_mlp(self.zeo_egnn(zeo))
        osda = self.osda_mlp(osda)

        # Sample from the prior distribution
        z = torch.randn(zeo.shape[0], self.z_dims).to(f'cuda:{zeo.get_device()}')
        z = torch.cat([z, zeo, osda], dim = -1)
        return self.decoder(z)
    
class GNN(torch.nn.Module):
    '''
    Custom GNN architecture that encodes zeolite graph 

    Args:
        n_node_feats, n_edge_feats: dimensions of graph features
        n_hidden: dimension of hidden
        p_drop: probability of dropout
        n_heads: number of attention heads in GATConv
    '''

    def __init__(self, n_node_feats, n_hidden, p_drop=0.0, n_heads=1, zeo_h_dims=64, pool_method='mean'):
        super(GNN, self).__init__()
        # torch.manual_seed(0)
        self.conv1 = GATConv(n_node_feats, n_hidden, heads=n_heads)
        self.conv2 = GATConv(n_hidden*n_heads, n_hidden, heads=n_heads)
        self.conv3 = GATConv(n_hidden*n_heads, n_hidden, heads=n_heads)

        self.mlp1 = nn.Linear(n_hidden*n_heads, n_hidden)
        self.mlp2 = nn.Linear(n_hidden, zeo_h_dims)

        self.act = nn.LeakyReLU()

        self.p_drop = p_drop
        self.pool_method = pool_method

    def forward(self, x, edge_index, edge_attr, batch):
        # Obtain node embeddings
        x = self.conv1(x, edge_index, edge_attr)
        x = self.act(x)
        x = F.dropout(x, p=self.p_drop, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)

        # Extra layer
        x = self.act(x)
        x = F.dropout(x, p=self.p_drop, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)

        # Readout layer
        if self.pool_method == 'mean':
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        elif self.pool_method == 'add':
            x = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        elif self.pool_method == 'max':
            x = global_max_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.mlp1(x)
        x = self.act(x)
        x = F.dropout(x, p=self.p_drop, training=self.training)
        x = self.mlp2(x)

        return x

class CVAE_GNN(nn.Module):
    def __init__(self, z_dims=64, zeo_h_dims=64, osda_feat_dims=14, osda_h_dims=64, syn_dims=12, n_node_feats=2, n_hidden=128, p_drop=0.0, n_heads=3, pool_method='mean'):
        super().__init__()
        self.z_dims   = z_dims
        self.osda_feat_dims = osda_feat_dims
        self.zeo_h_dims = zeo_h_dims
        self.osda_h_dims = osda_h_dims
        self.syn_dims = syn_dims
        self.n_node_feats = n_node_feats
        self.n_hidden = n_hidden
        self.p_drop = p_drop
        self.n_heads = n_heads
        self.pool_method = pool_method
        
        self.zeo_gnn = GNN(n_node_feats=self.n_node_feats, n_hidden=self.n_hidden, p_drop=self.p_drop, n_heads=self.n_heads, zeo_h_dims=self.zeo_h_dims, pool_method=self.pool_method)

        self.osda_mlp = nn.Sequential(nn.Linear(self.osda_feat_dims, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, self.osda_h_dims),
                                    )
        
        self.encoder = nn.Sequential(nn.Linear(self.syn_dims+self.zeo_h_dims+self.osda_h_dims, 4096),
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

        self.decoder = nn.Sequential(nn.Linear(self.z_dims+self.zeo_h_dims+self.osda_h_dims, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 2048),
                                     nn.ReLU(),
                                     nn.Linear(2048, 4096),
                                     nn.ReLU(),
                                     nn.Linear(4096, self.syn_dims),
                                     nn.Sigmoid(),
                                    )

    def forward(self, x, zeo, osda):
        
        # Encode zeolite and osda
        zeo = self.zeo_gnn(zeo.x, zeo.edge_index, zeo.edge_vec, zeo.batch)
        osda = self.osda_mlp(osda)

        # Concatenate zeolite and osda to the input
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

        # Encode zeolite and osda
        zeo = self.zeo_gnn(zeo.x, zeo.edge_index, zeo.edge_vec, zeo.batch)
        osda = self.osda_mlp(osda)

        # Sample from the prior distribution
        z = torch.randn(zeo.shape[0], self.z_dims).to(f'cuda:{zeo.get_device()}')
        z = torch.cat([z, zeo, osda], dim = -1)
        return self.decoder(z)