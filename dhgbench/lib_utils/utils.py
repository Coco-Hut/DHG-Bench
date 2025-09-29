import random
import numpy as np
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_add
from torch_geometric.utils import dropout_adj

def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]

def mean_std_metrics(metrics):
    
    metrics = np.array(metrics)
    metrics_dim = metrics.shape[-1]
    metrics = metrics.reshape(-1, metrics_dim) 
    
    metrics_mean = list(np.mean(metrics, axis=0))
    metrics_std = list(np.std(metrics, axis=0))
    
    return metrics_mean, metrics_std

def result_printer(metrics,name):
    
    metrics_dim = np.array(metrics).shape[-1]
    metrics_mean,metrics_std = mean_std_metrics(metrics)
    
    if metrics_dim == 1:
        print(f'{name}: {metrics_mean[0]:.4f}+-{metrics_std[0]:.2f}')
    else:
        print(f'train_{name}: {metrics_mean[0]:.4f}+-{metrics_std[0]:.2f}, val_{name}: {metrics_mean[1]:.4f}+-{metrics_std[1]:.2f}, test_{name}: {metrics_mean[2]:.4f}+-{metrics_std[2]:.2f}')

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# ------------ hypergraph add_self_loop ----------------------------------

def relabel_hyperedge_index(hyperedge_index):

    node_idx = hyperedge_index[0]
    hyperedge_idx = hyperedge_index[1]

    unique_ids, new_ids = torch.unique(hyperedge_idx, return_inverse=True)

    new_hyperedge_index = torch.stack([node_idx, new_ids], dim=0)

    return new_hyperedge_index, unique_ids

def add_self_loop_hyperedges(hyperedge_index: torch.Tensor, num_nodes: int):

    device = hyperedge_index.device

    num_existing_edges = hyperedge_index[1].max().item() + 1  

    self_loop_node_indices = torch.arange(num_nodes, device=device)
    self_loop_edge_indices = torch.arange(num_existing_edges, num_existing_edges + num_nodes, device=device)

    self_loop_index = torch.stack([self_loop_node_indices, self_loop_edge_indices], dim=0)  

    new_hyperedge_index = torch.cat([hyperedge_index, self_loop_index], dim=1)

    return new_hyperedge_index

# ------------ hypergraph/graph augmentation methods ----------------------

def drop_features(x: Tensor, p: float):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def filter_incidence(row: Tensor, col: Tensor, hyperedge_attr: OptTensor, mask: Tensor):
    return row[mask], col[mask], None if hyperedge_attr is None else hyperedge_attr[mask]

def drop_incidence(hyperedge_index: Tensor, p: float = 0.2):
    if p == 0.0:
        return hyperedge_index
    
    row, col = hyperedge_index
    mask = torch.rand(row.size(0), device=hyperedge_index.device) >= p
    
    row, col, _ = filter_incidence(row, col, None, mask)
    hyperedge_index = torch.stack([row, col], dim=0)
    return hyperedge_index

def drop_nodes(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_nodes, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    H[drop_idx, :] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index

def drop_hyperedges(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_edges, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    H[:, drop_idx] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index

def valid_node_edge_mask(hyperedge_index: Tensor, num_nodes: int, num_edges: int):
    ones = hyperedge_index.new_ones(hyperedge_index.shape[1])
    Dn = scatter_add(ones, hyperedge_index[0], dim=0, dim_size=num_nodes)
    De = scatter_add(ones, hyperedge_index[1], dim=0, dim_size=num_edges)
    node_mask = Dn != 0
    edge_mask = De != 0
    return node_mask, edge_mask

def common_node_edge_mask(hyperedge_indexs: list[Tensor], num_nodes: int, num_edges: int):
    hyperedge_weight = hyperedge_indexs[0].new_ones(num_edges)
    node_mask = hyperedge_indexs[0].new_ones((num_nodes,)).to(torch.bool)
    edge_mask = hyperedge_indexs[0].new_ones((num_edges,)).to(torch.bool)

    for index in hyperedge_indexs:
        Dn = scatter_add(hyperedge_weight[index[1]], index[0], dim=0, dim_size=num_nodes)
        De = scatter_add(index.new_ones(index.shape[1]), index[1], dim=0, dim_size=num_edges)
        node_mask &= Dn != 0
        edge_mask &= De != 0
    return node_mask, edge_mask

def hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask, edge_mask):
    if node_mask is None and edge_mask is None:
        return hyperedge_index

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    if node_mask is not None and edge_mask is not None:
        masked_hyperedge_index = H[node_mask][:, edge_mask].to_sparse().indices()
    elif node_mask is None and edge_mask is not None:
        masked_hyperedge_index = H[:, edge_mask].to_sparse().indices()
    elif node_mask is not None and edge_mask is None:
        masked_hyperedge_index = H[node_mask].to_sparse().indices()
    return masked_hyperedge_index

