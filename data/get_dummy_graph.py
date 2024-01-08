import torch
import torch_scatter
import torch_geometric
import torch_geometric.data
from torch_geometric.utils import to_dense_adj

def get_dummy_graph_for_dense_amor(code = 'Dense/Amorphous'):
    '''
    Returns a dummy graph for dense/amorphous

    Args:
    Code: str. Name of the zeolite
    '''

    n_nodes = 2
    n_edges = 2
    n_node_features = 2
    n_edge_features = 3

    x = torch.zeros((n_nodes, n_node_features), dtype = torch.float) # ZEROS
    edge_index = torch.tensor([
                                [0, 1],
                                [1, 0]
                                ])
    edge_vec = torch.zeros((n_edges, n_edge_features), dtype = torch.float)
    data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_vec=edge_vec, code=code) # IZA code
    
    return data

def get_dummy_graph_for_zeo(code):
    '''
    Returns a dummy graph for zeolite with missing CIF file

    Args:
    Code: str. Name of the zeolite
    '''

    n_nodes = 2
    n_edges = 2
    n_node_features = 2
    n_edge_features = 3

    x = torch.ones((n_nodes, n_node_features), dtype = torch.float) # ONES
    edge_index = torch.tensor([
                                [0, 1],
                                [1, 0]
                                ])
    edge_vec = torch.zeros((n_edges, n_edge_features), dtype = torch.float)
    data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_vec=edge_vec, code=code) # IZA code
    
    return data