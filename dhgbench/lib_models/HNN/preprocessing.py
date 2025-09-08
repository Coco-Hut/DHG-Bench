import time
import os
import os.path as osp
import random
import copy
import torch
import torch.nn.functional as F
import torch_sparse
import torch_scatter
import numpy as np
from collections import defaultdict
import dask
from dask.diagnostics import ProgressBar
from dask import delayed
import scipy.sparse as sp 
from lib_models.HNN.utils import symnormalise,ssm2tst,row_normalize,create_coo_from_edge_index
from typing import Optional
from tqdm import tqdm
import os 
from torch_geometric.utils import remove_self_loops

def algo_preprocessing(data,args):

    if args.method =='LEGCN':
        data = legcn_preprocessing(data,args)  # line_expansion
    elif args.method in ['HNHN']:
        data = generate_HNHN_norm(data,args)
    elif args.method in ['PhenomNNS','PhenomNN']:
        data = phenomNN_preprocessing(data,args)
    elif args.method == 'HJRL':
        data = hjrl_preprocessing(data,args)
    elif args.method == 'TMPHN':
        data = tmphn_preprocessing(data,args) 
    elif args.method == 'DPHGNN':
        data = dphgnn_preprocessing(data,args) 
    elif args.method == 'EHNN':
        data = ehnn_preprocessing(data,args)
    elif args.method in ['PlainUnigencoder']:
        data =uni_expansion(data,args)
    elif args.method == 'HyperGT':
        data = hypergt_preprocessing(data,args)
    else:
        pass
    
    return data

# ------------ Hypergraph Laplacian ----------------------

def hgnn_laplacian(H,type='full'):

    if type=="rw":
        # 1. H x B x H^{T}
        colsum = np.array(H.sum(0),dtype='float')
        r_inv_sqrt = np.power(colsum, -1).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        B = sp.diags(r_inv_sqrt)

        A_beta = H.dot(B).dot(H.transpose())
        I = sp.eye(A_beta.shape[0])
        A_beta+=I
        
        rowsum = np.array(A_beta.sum(1),dtype='float')
        r_inv_sqrt = np.power(rowsum, -1.0).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        D = sp.diags(r_inv_sqrt)
        
        A_beta = D.dot(A_beta)
        
        L = I-A_beta

    else:
        # 1. H x B x H^{T}
        colsum = np.array(H.sum(0),dtype='float')
        r_inv_sqrt = np.power(colsum, -1).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        B = sp.diags(r_inv_sqrt)

        A_beta = H.dot(B).dot(H.transpose())
        I = sp.eye(A_beta.shape[0])
        A_beta+=I

        # 2. D^{-1/2} x H x B x H^{T} x D^{1/2}
        rowsum = np.array(A_beta.sum(1),dtype='float')
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        D = sp.diags(r_inv_sqrt)

        A_beta = D.dot(A_beta).dot(D)
        
        if type=='sym':
            I = sp.eye(A_beta.shape[0])
            L=I-A_beta
        elif type=='full':
            L=A_beta
    
    return L

def Lapalacian_generator(data,args):
    H = create_coo_from_edge_index(data.hyperedge_index) 
    L_hgnn,L_sym,L_rw = hgnn_laplacian(H,type='full'),hgnn_laplacian(H,type='sym'),hgnn_laplacian(H,type='rw')
    return ssm2tst(L_hgnn).to(args.device),ssm2tst(L_sym).to(args.device),ssm2tst(L_rw).to(args.device)

# ------------ clique expansion methods ----------------------

def clique_expansion(data):
    H = create_coo_from_edge_index(data.hyperedge_index)
    clique_unorm=H.dot(H.transpose())
    clique_unorm.sum_duplicates() 
    src_idx,dst_idx=clique_unorm.nonzero()
    clique_edge_index = torch.tensor(np.vstack((src_idx, dst_idx)), dtype=torch.long)
    return clique_edge_index

# ------------ star expansion methods ----------------------

def star_expansion(data, num_nodes=None, num_hyperedges=None):

    nodes, hyperedges = data.hyperedge_index  

    if num_nodes is None:
        num_nodes = nodes.max().item() + 1  
    
    if num_hyperedges is None:
        num_hyperedges = hyperedges.max().item() + 1  

    hyperedge_offset = num_nodes  
    hyperedge_nodes = hyperedges + hyperedge_offset

    edge_index = torch.stack([nodes, hyperedge_nodes], dim=0)

    v_mask = torch.cat([torch.ones(num_nodes, dtype=torch.bool), torch.zeros(num_hyperedges, dtype=torch.bool)])

    return edge_index, v_mask

# ------------- DPHGNN preprocessing utils ----------------------

def dphgnn_preprocessing(data,args):
    # 1. clique expansion
    A_cg,L_cg = clique_preprocessing(data,args)
    
    # 2. hypergcn expansion
    A_hyp,L_hyp = hypergcn_preprocessing(data,args)
    
    # 3. star expansion
    A_sg,L_sg,v_mask = star_preprocessing(data,args)
    
    data.A_cg,data.A_hyp,data.A_sg,data.L_cg,data.L_hyp,data.L_sg,data.v_mask = A_cg,A_hyp,A_sg,L_cg,L_hyp,L_sg,v_mask
    
    data.L_hgnn,data.L_sym,data.L_rw = Lapalacian_generator(data,args)
    
    return data 

def gen_laplacian(A):
    rowsum = np.array(A.sum(1),dtype='float').flatten()
    D = sp.diags(rowsum)
    L=D-A
    return L

def star_preprocessing(data,args):
    star_graph,v_mask = star_expansion(data)
    star_graph_coo = create_coo_from_edge_index(star_graph,is_star=True) 
    
    A_sg=row_normalize(star_graph_coo)
    I = sp.eye(A_sg.shape[0])
    A_sg += I 
    
    L_sg = gen_laplacian(star_graph_coo)
    
    return ssm2tst(A_sg).to(args.device), ssm2tst(L_sg).to(args.device),v_mask

def clique_preprocessing(data,args):
    
    clique_graph = clique_expansion(data) 
    clique_graph_coo = create_coo_from_edge_index(clique_graph) 
    
    A_cg=row_normalize(clique_graph_coo)
    I = sp.eye(A_cg.shape[0])
    A_cg += I # (I+D_{v}^{-1}A_{c})
    
    L_cg = gen_laplacian(clique_graph_coo)
    return ssm2tst(A_cg).to(args.device),ssm2tst(L_cg).to(args.device)

def hypergcn_preprocessing(data,args):

    data_copy = copy.deepcopy(data)
    data_copy.to('cpu')
    He_dict = get_HyperGCN_He_dict(data_copy)
    A = hypergcn_expansion(V=data_copy.x.shape[0], E=He_dict, X=data_copy.x, m=args.mediator) 
    A_hyp = symnormalise(A) 
    L_hyp = gen_laplacian(A)
    
    return ssm2tst(A_hyp).to(args.device),ssm2tst(L_hyp).to(args.device)

def hypergcn_expansion(V, E, X, m):
    """
    Approximates hypergraph Laplacian using clique expansion or mediator method.

    Args:
        V (int): number of vertices
        E (dict): hyperedge dictionary {hid: list of nodes}
        X (ndarray): node features
        m (bool): use mediator mode or not

    Returns:
        A (torch.sparse.Tensor): normalized adjacency matrix
    """
    edges = []
    weights = defaultdict(float)
    rv = np.random.rand(X.shape[1])
    X_proj = X @ rv  # shape: (n_nodes,)

    for hyperedge in E.values():

        hyperedge = list(hyperedge)
        proj = X_proj[hyperedge]
        s, i = np.argmax(proj), np.argmin(proj)
        Se, Ie = hyperedge[s], hyperedge[i]

        if m:
            c = max(2 * len(hyperedge) - 3, 1)

            edges.extend([[Se, Ie], [Ie, Se]])
            weights[(Se, Ie)] += 1.0 / c
            weights[(Ie, Se)] += 1.0 / c

            for mediator in hyperedge:
                if mediator != Se and mediator != Ie:
                    for u, v in [(Se, mediator), (Ie, mediator), (mediator, Se), (mediator, Ie)]:
                        edges.append([u, v])
                        weights[(u, v)] += 1.0 / c
        else:
            e = len(hyperedge)
            edges.extend([[Se, Ie], [Ie, Se]])
            weights[(Se, Ie)] += 1.0 / e
            weights[(Ie, Se)] += 1.0 / e

    return adjacency(edges, weights, V)

def adjacency(edges, weights, n):
    """
    computes an sparse adjacency matrix

    arguments:
    edges: list of pairs
    weights: dictionary of edge weights (key: tuple representing edge, value: weight on the edge)
    n: number of nodes

    returns: a scipy.sparse adjacency matrix with unit weight self loops for edges with the given weights
    """
    
    dictionary = {tuple(item): index for index, item in enumerate(edges)}
    edges = [list(itm) for itm in dictionary.keys()]   
    organised = []

    for e in edges:
        i,j = e[0],e[1]
        w = weights[(i,j)]
        organised.append(w)

    edges, weights = np.array(edges), np.array(organised)
    adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + sp.eye(n)

    A = sp.csr_matrix(adj, dtype=np.float32)
    return A

# ------------- PhenomNN preprocessing utils ----------------------

def phenomNN_expansion(H,args,norm_type='full'):

    if norm_type=="full":

        # 1. H x B x H^{T}
        colsum = np.array(H.sum(0),dtype='float')
        r_inv_sqrt = np.power(colsum, -1).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        B = sp.diags(r_inv_sqrt)

        A_beta = H.dot(B).dot(H.transpose())
        I = sp.eye(A_beta.shape[0])
        A_beta += I

        # 2. D^{-1/2} x H x B x H^{T} x D^{1/2}
        rowsum = np.array(A_beta.sum(1),dtype='float')
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        D = sp.diags(r_inv_sqrt)

        A_beta = D.dot(A_beta).dot(D)

        rowsum_norm = np.array(A_beta.sum(1))
        rowsum_norm_flat = rowsum_norm.flatten()
        D_beta = sp.diags(rowsum_norm_flat)

    elif norm_type=="node":

        A_beta=H.dot(H.transpose())
        I = sp.eye(A_beta.shape[0])   
        A_beta+=I
        
        rowsum = np.array(A_beta.sum(1),dtype='float')
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        D = sp.diags(r_inv_sqrt)

        A_beta = D.dot(A_beta).dot(D)

        rowsum_norm = np.array(A_beta.sum(1))
        rowsum_norm_flat = rowsum_norm.flatten()
        D_beta = sp.diags(rowsum_norm_flat)

    return ssm2tst(A_beta).to(args.device),ssm2tst(D_beta).to(args.device),ssm2tst(I).to(args.device) 

def phenomNN_preprocessing(data,args):

    H = create_coo_from_edge_index(data.hyperedge_index)

    if args.method == 'PhenomNN':
        A_beta,D_beta,I=phenomNN_expansion(H,args,norm_type="node")
        A_gamma,D_gamma,_=phenomNN_expansion(H,args,norm_type="full")
        H_expansion=[A_beta,A_gamma]
        G=[D_beta,D_gamma,I]
    else:
        A_beta,D_beta,I=phenomNN_expansion(H,args,norm_type="node")
        A_gamma,D_gamma,_=phenomNN_expansion(H,args,norm_type="full")
        A= args.lam0 * A_beta +args.lam1 *  A_gamma
        D=args.lam0 * D_beta +args.lam1 *  D_gamma
        H_expansion=A 
        G=[D,I]

    data.adj, data.G = H_expansion, G
    
    return data

# ------------- HNHN preprocessing utils ----------------------

def generate_HNHN_norm(data, args):
    """
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """

    edge_index=copy.deepcopy(data.hyperedge_index)
    edge_index = edge_index.to('cpu')

    # Construct incidence matrix H of size (num_nodes, num_hyperedges) from edge_index = [V;E]
    ones = torch.ones(edge_index.shape[1], device=edge_index.device)

    alpha = args.HNHN_alpha
    beta = args.HNHN_beta

    # the degree of the node
    DV = torch_scatter.scatter_add(ones, edge_index[0], dim=0)
    # the degree of the hyperedge
    DE = torch_scatter.scatter_add(ones, edge_index[1], dim=0)

    # alpha part
    D_e_alpha = DE ** alpha
    D_e_alpha[D_e_alpha == float("inf")] = 0 
    D_v_alpha = torch_scatter.scatter_add(DE[edge_index[1]], edge_index[0], dim=0)

    # beta part
    D_v_beta = DV ** beta
    D_v_beta[D_v_beta == float("inf")] = 0 
    D_e_beta = torch_scatter.scatter_add(DV[edge_index[0]], edge_index[1], dim=0)

    D_v_alpha_inv = 1.0 / D_v_alpha
    D_v_alpha_inv[D_v_alpha_inv == float("inf")] = 0

    D_e_beta_inv = 1.0 / D_e_beta
    D_e_beta_inv[D_e_beta_inv == float("inf")] = 0

    data.D_e_alpha,data.D_v_alpha_inv,data.D_v_beta,data.D_e_beta_inv = D_e_alpha.float().to(args.device),D_v_alpha_inv.float().to(args.device),D_v_beta.float().to(args.device),D_e_beta_inv.float().to(args.device)
    
    return data

# ------------- Unigencoder preprocessing utils ----------------------------

def uni_expansion(data, args):
    
    threshold, norm_type, init_val, init_type = args.threshold, args.norm_type, args.init_val, args.init_type
    
    V, E = data.hyperedge_index
    # print(data.edge_index.shape)
    edge_dict = {}
    for i in range(data.hyperedge_index.shape[1]):
        if E[i].item() not in edge_dict:
            edge_dict[E[i].item()] = []
        edge_dict[E[i].item()].append(V[i].item())

    N_vertex = V.max() + 1
    N_hyperedge = data.hyperedge_index.shape[1]
    N_hy = len(edge_dict)
    self_set = set()
    # print(len(edge_dict), N_hyperedge)
    for key, val in edge_dict.items():
        if len(val) == 1:
            self_set.add(val[0])
    # print(len(self_set))
    if len(self_set) < N_vertex:
        # print(len(self_set))
        count = 0
        for i in range(N_vertex):
            if i not in self_set:
                edge_dict[N_hy + count] = []
                edge_dict[N_hy+count].append(i)
        count+=1

    node_num_egde = {}
    for key, val in edge_dict.items():
        for v in val:
            if v not in node_num_egde:
                node_num_egde[v] = 0
            else:
                node_num_egde[v] += 1

    pv_rows = []
    pv_cols = []

    n_count = 0
    if threshold >= -1.0:
        def cosine_similarity_dense_small(x):
            norm = F.normalize(x, p=2., dim=-1)  # [N, out_channels]
            sim = norm.mm(norm.t())
            return sim
        sim = cosine_similarity_dense_small(data.x)
        # print(sim.shape)
        for i in range(len(edge_dict)):
            neighbor_indices = np.array(edge_dict.get(i, []), dtype=np.int32)
            res_idx = torch.tensor(neighbor_indices)
            # sim_mean = 0
            if len(neighbor_indices) > 1:
                # continue
                idx = torch.tensor(neighbor_indices)
                new_sim = torch.index_select(torch.index_select(sim, 0, idx), 1, idx)
                new_sim[torch.eye(len(idx)).bool()] = 0.0
                # print(torch.sum(new_sim))
                sim_mean = torch.sum(new_sim) / (len(idx)*(len(idx)-1))

                if sim_mean > threshold:
                    for q, p in enumerate(res_idx):
                        pv_rows.append(n_count)
                        # pv_rows.append(i)
                        pv_cols.append(p)
                        if q == (len(res_idx)-1):
                            n_count += 1
            else:
                for q, p in enumerate(res_idx):
                    pv_rows.append(n_count)
                    # pv_rows.append(i)
                    pv_cols.append(p)
                    # if q == (len(res_idx)-1):
                    n_count += 1
        n_count2 = n_count
    elif -1 > threshold >= -2:

        n_count2 = n_count
        for i in range(len(edge_dict)):
            res_idx = np.array(edge_dict.get(i, []), dtype=np.int32)

            for q, p in enumerate(res_idx):
                pv_rows.append(n_count2)
                pv_cols.append(p)
                # if q == (len(res_idx)-1):
                n_count2 += 1
    else:
        n_count2 = n_count
        pv_init_val = []
        for i in range(len(edge_dict)):
            res_idx = np.array(edge_dict.get(i, []), dtype=np.int32)
            for q, p in enumerate(res_idx):
                if len(res_idx) == 1:
                    if res_idx[0] in node_num_egde:
                        if init_type == 1:
                            pv_init_val.append(node_num_egde[res_idx[0]]*init_val)
                        else:
                            pv_init_val.append(init_val)
                        # print()
                    else:
                        pv_init_val.append(1)
                else:
                    pv_init_val.append(1)
                pv_rows.append(n_count2)
                pv_cols.append(p)
                if q == (len(res_idx)-1):
                    n_count2 += 1
    pv_rows = torch.tensor(pv_rows)
    pv_cols = torch.tensor(pv_cols)
    pv_indices = torch.stack([pv_rows, pv_cols], dim=0)
    pv_values = torch.tensor(pv_init_val, dtype=torch.float32)

    Pv = torch.sparse_coo_tensor(pv_indices, pv_values, size=[n_count2, N_vertex])
    PvT = torch.sparse_coo_tensor(torch.stack([pv_cols, pv_rows], dim=0), pv_values, size=[N_vertex, n_count2])

    if norm_type == 0:
        Pv_col_sum = torch.sparse.sum(Pv, dim=1)
        Pv_diag_indices = Pv_col_sum.indices()[0]
        Pv_diag_values = Pv_col_sum.values()
        Pv_diag_values = torch.reciprocal(Pv_diag_values)
        Pv_diag = torch.sparse_coo_tensor(torch.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
                                        torch.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
        Pv_col_norm = torch.sparse.mm(Pv_diag, Pv)

        PvT_row_sum = torch.sparse.sum(PvT, dim=1)
        PvT_diag_indices = PvT_row_sum.indices()[0]
        PvT_diag_values = PvT_row_sum.values()
        PvT_diag_values = torch.reciprocal(PvT_diag_values)
        PvT_diag = torch.sparse_coo_tensor(torch.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
                                        torch.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))
        # print(PvT.shape, PvT_diag.shape)
        PvT_row_norm = torch.sparse.mm(PvT_diag, PvT)
    elif norm_type == 1:
        Pv_col_sum = torch.sparse.sum(Pv, dim=0)
        Pv_diag_indices = Pv_col_sum.indices()[0]
        Pv_diag_values = Pv_col_sum.values()
        Pv_diag_values = torch.reciprocal(Pv_diag_values)
        Pv_diag = torch.sparse_coo_tensor(torch.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
                                            torch.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
        # print(Pv.shape, Pv_diag.shape)
        Pv_col_norm = torch.sparse.mm(Pv, Pv_diag)

        PvT_row_sum = torch.sparse.sum(PvT, dim=1)
        PvT_diag_indices = PvT_row_sum.indices()[0]
        PvT_diag_values = PvT_row_sum.values()
        PvT_diag_values = torch.reciprocal(PvT_diag_values)
        PvT_diag = torch.sparse_coo_tensor(torch.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
                                            torch.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))
        # print(PvT.shape, PvT_diag.shape)
        PvT_row_norm = torch.sparse.mm(PvT_diag, PvT)
    elif norm_type == 2:
        Pv_col_sum = torch.sparse.sum(Pv, dim=0)
        Pv_diag_indices = Pv_col_sum.indices()[0]
        Pv_diag_values = Pv_col_sum.values()
        Pv_diag_values = torch.reciprocal(Pv_diag_values)
        Pv_diag = torch.sparse_coo_tensor(torch.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
                                            torch.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
        # print(Pv.shape, Pv_diag.shape)
        Pv_col_norm = torch.sparse.mm(Pv, Pv_diag)

        PvT_row_sum = torch.sparse.sum(PvT, dim=0)
        PvT_diag_indices = PvT_row_sum.indices()[0]
        PvT_diag_values = PvT_row_sum.values()
        PvT_diag_values = torch.reciprocal(PvT_diag_values)
        PvT_diag = torch.sparse_coo_tensor(torch.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
                                            torch.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))
        # print(PvT.shape, PvT_diag.shape)
        PvT_row_norm = torch.sparse.mm(PvT, PvT_diag)
    elif norm_type == 3:
        Pv_col_sum = torch.sparse.sum(Pv, dim=1)
        Pv_diag_indices = Pv_col_sum.indices()[0]
        Pv_diag_values = Pv_col_sum.values()
        Pv_diag_values = torch.reciprocal(Pv_diag_values)
        Pv_diag = torch.sparse_coo_tensor(torch.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
                                            torch.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
        # print(Pv.shape, Pv_diag.shape)
        Pv_col_norm = torch.sparse.mm(Pv_diag, Pv)

        PvT_row_sum = torch.sparse.sum(PvT, dim=0)
        PvT_diag_indices = PvT_row_sum.indices()[0]
        PvT_diag_values = PvT_row_sum.values()
        PvT_diag_values = torch.reciprocal(PvT_diag_values)
        PvT_diag = torch.sparse_coo_tensor(torch.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
                                            torch.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))

        PvT_row_norm = torch.sparse.mm(PvT, PvT_diag)
    else:
        Pv_col_norm = Pv
        PvT_row_norm = PvT

    print(Pv_col_norm.shape, data.x.shape)
    
    data.Pv,data.PvT = Pv_col_norm.to(args.device),PvT_row_norm.to(args.device)
    
    return data

#---------------------- LEGCN preprocessing utils -------------------------

def legcn_preprocessing(data,args):
    data.le_adj = line_expansion(data,args)
    return data

def line_expansion(data,args):
    hyperedge_index=copy.deepcopy(data.hyperedge_index)
    hyperedge_index = hyperedge_index.to('cpu')
    V, E = hyperedge_index
    E = E + V.max() # [V | E]
    num_ne_pairs = hyperedge_index.shape[1]
    V_plus_E = E.max() + 1
    L1 = torch.stack([torch.arange(V.shape[0], device=V.device), V], -1)
    L2 = torch.stack([torch.arange(E.shape[0], device=E.device), E], -1)
    L = torch.cat([L1, L2], -1) # [2, |V| + |E|]
    L_T = torch.stack([L[1], L[0]], 0) # [2, |V| + |E|]
    ones = torch.ones(L.shape[1], device=L.device)
    adj, value = torch_sparse.spspmm(L, ones, L_T, ones, num_ne_pairs, V_plus_E, num_ne_pairs, coalesced=True)
    adj, value = torch_sparse.coalesce(adj, value, num_ne_pairs, num_ne_pairs, op="add")
    return adj.to(args.device)

#---------------------- HJRL preprocessing utils -------------------------

def hjrl_preprocessing(data,args):
    
    hyperedge_index=copy.deepcopy(data.hyperedge_index)
    hyperedge_index = hyperedge_index.to('cpu')

    row = hyperedge_index[0].numpy()  
    col = hyperedge_index[1].numpy() 
    values = np.ones(len(row))  

    num_nodes = row.max() + 1 
    num_hyperedges = col.max() + 1  
    H_ini = sp.coo_matrix((values, (row, col)), shape=(num_nodes, num_hyperedges)).astype(np.float32) 

    HHT, H = normalize_sparse_hypergraph_symmetric(H_ini)
    HTH, HT = normalize_sparse_hypergraph_symmetric(H_ini.transpose())
    
    # 2. node/edge feature normalization
    X,E = precompute_node_edge_features(H_ini,data.x,args.device)
    
    HHT = ssm2tst(HHT).to(args.device)
    H = ssm2tst(H).to(args.device)
    HT = ssm2tst(HT).to(args.device)
    HTH = ssm2tst(HTH).to(args.device) 
    H_ini = torch.from_numpy(H_ini.toarray()).to(args.device)
    
    data.H_ini, data.H, data.HT, data.HHT, data.HTH, data.norm_X, data.norm_E = H_ini, H, HT, HHT, HTH, X, E

    return data

def normalize_sparse_hypergraph_symmetric(H):
    rowsum = np.array(H.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    D = sp.diags(r_inv_sqrt)

    colsum = np.array(H.sum(0),dtype='float')
    r_inv_sqrt = np.power(colsum, -1).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    B = sp.diags(r_inv_sqrt)

    Omega = sp.eye(B.shape[0])

    hx1 = D.dot(H).dot(Omega).dot(B).dot(H.transpose()).dot(D)
    hx2 = D.dot(H).dot(Omega).dot(B)

    return hx1, hx2

def precompute_node_edge_features(H_ini,X,device):

    X = copy.deepcopy(X)
    X = X.cpu()
    E_node = H_ini.toarray().shape[1]
    H_T = H_ini.toarray().T
    E = np.zeros((E_node, X.shape[1]))
    for i in range(E_node):
        flag = np.where(H_T[i])[0]
        flag = X[flag]
        flag = flag.sum(0)
        E[i] = flag
    E = np.where(E, 1, 0).astype(float)
    #X = normalize_features(sp.csr_matrix(X))
    #E = normalize_features(sp.csr_matrix(E))
    X = row_normalize(sp.csr_matrix(X))
    E = row_normalize(sp.csr_matrix(E))
    
    X = torch.from_numpy(X.toarray()).float().to(device)
    E = torch.from_numpy(E.toarray()).float().to(device)

    return X,E

#---------------------- HyperGCN preprocessing utils -------------------------

# functions for processing/checkning the edge_index
def get_HyperGCN_He_dict_old(data):
    # Assume edge_index = [V;E], sorted
    edge_index = np.array(data.hyperedge_index.cpu())
    """
    For each he, clique-expansion. Note that we allow the weighted edge.
    Note that if node pair (vi,vj) is contained in both he1, he2, we will have (vi,vj) twice in edge_index. (weighted version CE)
    We default no self loops so far.
    """
    # edge_index[1, :] = edge_index[1, :]-edge_index[1, :].min()
    He_dict = {}
    for he in np.unique(edge_index[1, :]):
        nodes_in_he = list(edge_index[0, :][edge_index[1, :] == he])
        He_dict[he.item()] = nodes_in_he

    return He_dict

def get_HyperGCN_He_dict(data):
    edge_index = np.array(data.hyperedge_index.cpu())
    He_dict = defaultdict(list)

    for node, he in zip(edge_index[0, :], edge_index[1, :]):
        He_dict[he.item()].append(node)

    return dict(He_dict)

def update(Se, Ie, mediator, weights, c):
    """
    updates the weight on {Se,mediator} and {Ie,mediator}
    """    
    
    if (Se,mediator) not in weights:
        weights[(Se,mediator)] = 0
    weights[(Se,mediator)] += float(1/c)

    if (Ie,mediator) not in weights:
        weights[(Ie,mediator)] = 0
    weights[(Ie,mediator)] += float(1/c)

    if (mediator,Se) not in weights:
        weights[(mediator,Se)] = 0
    weights[(mediator,Se)] += float(1/c)

    if (mediator,Ie) not in weights:
        weights[(mediator,Ie)] = 0
    weights[(mediator,Ie)] += float(1/c)

    return weights

#---------------------- TMPHN preprocessing utils -------------------------

def tmphn_preprocessing(data,args):

    hyperedge_index=copy.deepcopy(data.hyperedge_index)
    hyperedge_index = hyperedge_index.to('cpu')

    row = hyperedge_index[0].numpy()  
    col = hyperedge_index[1].numpy()  
    values = np.ones(len(row))  

    num_nodes = row.max() + 1  
    num_hyperedges = col.max() + 1 
    H = sp.coo_matrix((values, (row, col)), shape=(num_nodes, num_hyperedges)).astype(np.float32) 
    H=H.toarray()
    
    all_nodes = list(np.arange(data.x.shape[0])) 
    neig_dict = NeighborFinder(H, args).neig_for_targets(all_nodes)
    data.neig_dict = neig_dict
    data.all_idx = np.arange(data.x.shape[0]) 
    data.target_idx = np.arange(data.x.shape[0]) 
    
    return data

class NeighborFinder:
    """
    Given a set of target nodes, find their neighbors
    args:
    edge_dict: an edge dictionary where key = edge_id, value = a list of nodes in this edge
    M: the maximum number of neighbors to be sampled for each target node (order of the hypergraph)
    return :
    a neighbhorhood dictinoary where key = a targe node, value = a nested list contains its neighbors
    """
    def __init__(self, H, args):
        self.H = H
        self.E = H.shape[1]
        self.M = args.M
        edge_dict = self.from_H_to_dict()
        self.adj_lst = [list(edge_dict[edge_id]) for edge_id in range(len(edge_dict))]#the serch space
        # self.adj_lst = adj_lst
        
    def neig_for_targets(self, target_nodes):
        """
        use dask to search over the adj_lst to find neighbors for all target nodes
        return: 
        batch_dict: a dictionary maps target_nodes to their neighbors
        """
        print('NeighborFinder Search: ')
        neig_list = []
        for x in target_nodes:
            y = delayed(self.find_neigs_of_one_node)(x)
            neig_list.append(y)
        with ProgressBar():
            neig_lst = dask.compute(neig_list, num_workers=os.cpu_count()*2)
        
        neig_lst = sum(neig_lst, []) #un-neste 
        
        batch_dict = dict(zip(target_nodes, neig_lst))

        return batch_dict
        
    def find_neigs_of_one_node(self, target_node):
        neigs_of_node = []
        for edge in self.adj_lst:
            if target_node in edge:
                if len(edge) <= self.M:
                    neigs_of_node.append(list(edge))
                else:
                    edge_lst = list(edge)
                    tmp = copy.deepcopy(edge_lst)
                    tmp.remove(target_node)
                    random.seed(42)
                    neigs_of_node.append(random.sample(tmp, self.M - 1) + [target_node])      
        return neigs_of_node
    
    def from_H_to_dict(self):
        """
        Take the incidence matrix as input, produce the incidence dictionary
        that will be used in message passing.
        Input: 
        H: incidence matrix, (N, E)
        Output:
        inci_dic: incidence dictionary with key = edge id, value = set(incident node idx)
        """
        edges_lst = [set(np.nonzero(self.H[:,i])[0]) for i in range(self.E)] #all edges
        edge_idx_lst = list(np.arange(0, self.E, 1))
        edge_dict = dict(map(lambda i,j : (i,j) , edge_idx_lst, edges_lst))
        return edge_dict
    
    def from_csc_H_to_dict(self):
        """
        Take the incidence matrix as input, produce the incidence dictionary
        that will be used in message passing.
        Input: 
        H: incidence matrix that is stored in csc format, (N, E)
        Output:
        inci_dic: incidence dictionary with key = edge id, value = set(incident node idx)
        
        Note: this function is used for the business datasets
        """
        edge_dict = {}
        for col in range(self.H.shape[1]): # go through each hyperedge
            nonzero_rows = list(self.H[:, col].indices) #get the nodes
            edge_dict[col] = nonzero_rows
        return edge_dict

#---------------------- EHNN preprocessing utils -------------------------

def ehnn_preprocessing(data,args,folder='lib_ehnn_cache'):
    
    src_data = copy.deepcopy(data)
    data = data.to('cpu')
    
    # build memory-efficiently for yelp and walmart-trips-100
    if args.dname not in ["yelp", "walmart-trips-100"]:
        original_edge_index = data.hyperedge_index
        data = ConstructH(data)  # [|V|, |E|]
        incidence_d = (
            torch.tensor(data.hyperedge_index, dtype=torch.float32)
            .to_sparse(2)
            .coalesce()
            .to(args.device)
        )
        edge_orders_d = (
            torch.sparse.sum(incidence_d, 0).to_dense().long().to(args.device)
        )  # [|E|,]

        data.hyperedge_index = original_edge_index
        data = ConstructHSparse(data)
        num_nodes = data.x.shape[0]
        num_hyperedges = np.max(data.hyperedge_index[1]) + 1
        incidence_s = torch.sparse_coo_tensor(
            data.hyperedge_index,
            torch.ones(len(data.hyperedge_index[0])),
            (num_nodes, num_hyperedges),
            device=args.device,
        ).coalesce()
        edge_orders_s = (
            torch.sparse.sum(incidence_s, 0).to_dense().long().to(args.device)
        )  # [|E|,]

        assert (incidence_d.indices() - incidence_s.indices() == 0).all()
        assert (incidence_d.values() - incidence_s.values() == 0).all()

        incidence = incidence_d
        edge_orders = edge_orders_d
    else:
        data = ConstructHSparse(data)
        num_nodes = data.x.shape[0]
        num_hyperedges = np.max(data.hyperedge_index[1]) + 1
        incidence = (
            torch.sparse_coo_tensor(
                data.hyperedge_index,
                torch.ones(len(data.hyperedge_index[0])),
                (num_nodes, num_hyperedges),
            )
            .coalesce()
            .to(args.device)
        )
        edge_orders = torch.sparse.sum(incidence, 0).to_dense().long().to(args.device)

    os.makedirs(f"./{folder}", exist_ok=True)
    
    if not osp.isfile(f"./{folder}/{args.dname}.pt") or args.task_type=='hg_cls':
        print(f"preprocessing {args.dname}")
        overlaps = None
        masks = None
        n_overlaps = None
        prefix_normalizer = (torch.sparse.sum(incidence, 0).to_dense().to(args.device))  # [|E|,]
        prefix_normalizer = prefix_normalizer.masked_fill_(prefix_normalizer == 0, 1e-5)
        normalizer = None
        suffix_normalizer = (torch.sparse.sum(incidence, 1).to_dense().to(args.device))  # [|V|,]
        suffix_normalizer = suffix_normalizer.masked_fill_(suffix_normalizer == 0, 1e-5)

        if args.dname not in ["yelp", "walmart-trips-100"]:
            # chunked mask computation
            mask_dict_chunk = build_mask_chunk(incidence, args.device, args.chunk_size)  # Dict(overlap: [|E|, |E|] sparse)
            overlaps_chunk, masks_chunk = list(mask_dict_chunk.keys()), list(mask_dict_chunk.values())
            overlaps_chunk = torch.tensor(overlaps_chunk, dtype=torch.long, device=args.device)  # [|overlaps|,]
            masks_chunk = torch.stack(masks_chunk, dim=0).coalesce()  # [|overlaps|, |E|, |E|]]

            # correctness check with non-chunked masks
            mask_dict = build_mask(incidence, args.device)  # Dict(overlap: [|E|, |E|] sparse)
            overlaps, masks = list(mask_dict.keys()), list(mask_dict.values())
            overlaps = torch.tensor(overlaps, dtype=torch.long, device=args.device)  # [|overlaps|,]
            masks = torch.stack(masks, dim=0).coalesce()  # [|overlaps|, |E|, |E|]]
            assert (masks.indices() - masks_chunk.indices() == 0).all()
            assert (masks.values() - masks_chunk.values() == 0).all()
            assert (overlaps - overlaps_chunk == 0).all()

            masks = masks_chunk
            overlaps = overlaps_chunk
            n_overlaps = len(overlaps)

            normalizer = (torch.sparse.sum(masks, 2).to_dense().unsqueeze(-1))  # [|overlaps|, |E|, 1]
            normalizer = normalizer.masked_fill_(normalizer == 0, 1e-5)

        ehnn_cache = dict(
            incidence=incidence,
            edge_orders=edge_orders,
            overlaps=overlaps,
            n_overlaps=n_overlaps,
            prefix_normalizer=prefix_normalizer,
            suffix_normalizer=suffix_normalizer,
        )

        torch.save(ehnn_cache, f"./{folder}/{args.dname}.pt")
        print(f"saved ehnn_cache for {args.dname}")
    else:
        ehnn_cache = torch.load(f"./{folder}/{args.dname}.pt")
    print(f"number of mask channels: {ehnn_cache['n_overlaps']}")
    
    src_data.ehnn_cache = ehnn_cache
    
    return src_data

@torch.no_grad()
def build_mask_chunk(incidence, device, chunk_size = 1000):
    tic = time.time()
    stat_overlap_num = {}  # key: overlap, value: number of edges
    incidence = incidence.to(device)
    N = incidence.size(0)
    E = incidence.size(1)
    chunk_begin_index = 0
    output = torch.sparse_coo_tensor(
        torch.empty([2, 0]), [], [E, E], device=device
    ).coalesce()
    target_indices = incidence.indices()[
        [1, 0]
    ]  # caution: indices of transposed incidence matrix
    row_indices = target_indices[0]  # [|E|,]
    target_values = incidence.values()  # [|E|,]

    while chunk_begin_index < E:
        chunk_end_index = min(chunk_begin_index + chunk_size, incidence.size(1))
        chunk_mask = torch.logical_and(
            row_indices >= chunk_begin_index, row_indices < chunk_end_index
        )  # [|chunk|]
        chunk_matrix = torch.sparse_coo_tensor(
            target_indices[:, chunk_mask],
            target_values[chunk_mask],
            [E, N],
            device=device,
        ).coalesce()  # [2, |chunk|]
        del chunk_mask
        print(
            f"chunk_begin_index: {chunk_begin_index}/{E}, "
            f"output: {output.indices().size(1)} nnz, "
            f"chunk_matrix: {chunk_matrix.indices().size(1)} nnz"
        )
        chunk_output = torch.sparse.mm(chunk_matrix, incidence)
        output = (output + chunk_output).coalesce()
        del chunk_matrix
        chunk_begin_index += chunk_size

    overlap_mask_indices = defaultdict(lambda: [[], []])
    for overlap, row_idx, col_idx in zip(
        output.values(), output.indices()[0], output.indices()[1]
    ):
        overlap = int(overlap.item())
        overlap_mask_indices[overlap][0].append(row_idx.item())
        overlap_mask_indices[overlap][1].append(col_idx.item())

    overlap_masks = {}
    for overlap, mask_indices in overlap_mask_indices.items():
        overlap_masks[overlap] = torch.sparse_coo_tensor(
            mask_indices, torch.ones(len(mask_indices[0])), (E, E), device=device
        )
    print(f"Took {time.time() - tic} seconds")
    return overlap_masks

@torch.no_grad()
def build_mask(incidence, device):
    tic = time.time()
    overlap_masks = {}
    # stat_overlap_num = {}  # Dict(overlap: number of edges)

    incidence = incidence.to(device)

    E = incidence.size(1)
    pre_overlap = torch.sparse.mm(incidence.t(), incidence).coalesce()  # [|E|, |E|]
    # stat_num_entry = len(pre_overlap.indices()[0])
    overlap_mask_indices = defaultdict(lambda: [[], []])

    # Generate Overlap masks - Not made for overlap mask of overlap 0
    for overlap, row_idx, col_idx in zip(
        pre_overlap.values(), pre_overlap.indices()[0], pre_overlap.indices()[1]
    ):
        overlap = int(overlap.item())
        overlap_mask_indices[overlap][0].append(row_idx.item())
        overlap_mask_indices[overlap][1].append(col_idx.item())

    for overlap, mask_indices in overlap_mask_indices.items():
        overlap_masks[overlap] = torch.sparse_coo_tensor(
            mask_indices, torch.ones(len(mask_indices[0])), (E, E), device=device
        )
    print(f"Took {time.time() - tic} seconds")

    return overlap_masks

def ConstructH(data):
    """
    Construct incidence matrix H of size (num_nodes,num_hyperedges) from edge_index = [V;E]
    """
    edge_index = np.array(data.hyperedge_index)
    num_nodes = data.x.shape[0]
    num_hyperedges = np.max(edge_index[1]) - np.min(edge_index[1]) + 1
    H = np.zeros((num_nodes, num_hyperedges))
    cur_idx = 0
    for he in np.unique(edge_index[1]):
        nodes_in_he = edge_index[0][edge_index[1] == he]
        H[nodes_in_he, cur_idx] = 1.0
        cur_idx += 1

    data.hyperedge_index = H
    return data

def ConstructHSparse(data):
    """
    Construct incidence matrix H of size (num_nodes,num_hyperedges) from edge_index = [V;E]
    """
    edge_index = np.array(data.hyperedge_index)
    num_nodes = data.x.shape[0]
    num_hyperedges = np.max(edge_index[1]) - np.min(edge_index[1]) + 1
    assert np.min(edge_index[1]) == 0 and np.max(edge_index[1]) == num_hyperedges - 1
    data.hyperedge_index = edge_index
    return data

#---------------------- HyperGT preprocessing utils -------------------------

def hypergt_preprocessing(data, args, add_reverse=False, add_token_loops=True):
    """
      output: 
      - data.num_tokens = n + m
      - data.H:    [n, m] (float32, device)
      - data.adjs: [edge_index_token] (long, device)
      - data.x:    [n+m, d]
    """
    dev = torch.device(args.device)

    ei = data.hyperedge_index.to(torch.long).to(dev)   # [2, E]
    v  = ei[0]                                         # node id
    e0 = ei[1]                                         # hyperedge id
    n  = int(getattr(data, 'num_nodes', data.x.size(0)))

    e_min = int(e0.min().item())
    e     = e0 - e_min                                 # 0..m-1
    m     = int(e.max().item()) + 1
    N     = n + m

    # ---- 2) construct H: [n, m] (float32, device) ----
    vals = torch.ones(e.size(0), dtype=torch.float32, device=dev)
    H = torch.sparse_coo_tensor(
        indices=torch.stack([v, e], dim=0),            # [2, E]
        values=vals,
        size=(n, m),
        dtype=torch.float32,
        device=dev
    ).coalesce()

    # ---- 3) construct token adjs[0]（V->[n..n+m-1]）----
    e_off = e + n                                   
    edge_v2e = torch.stack([v, e_off], dim=0)          # [2, E]
    edge_v2e, _ = remove_self_loops(edge_v2e)

    if add_reverse:
        edge_e2v = torch.stack([e_off, v], dim=0)      # [2, E]
        edge_index = torch.cat([edge_v2e, edge_e2v], dim=1)  # [2, 2E]
    else:
        edge_index = edge_v2e

    if add_token_loops:
        loops = torch.arange(N, device=dev, dtype=torch.long)
        loop_edge = torch.stack([loops, loops], dim=0)       # [2, N]
        edge_index = torch.cat([edge_index, loop_edge], dim=1)

    edge_index = edge_index.to(torch.long).to(dev)

    # ---- 4) x -> [n, m]
    x = data.x
    if x.device != dev:
        x = x.to(dev)
    if x.size(0) == n:
        pad = torch.zeros((m, x.size(1)), dtype=x.dtype, device=dev)
        x = torch.cat([x, pad], dim=0)                       # [n+m, F]
    elif x.size(0) != N:
        raise ValueError(f"data.x.shape[0]={x.size(0)} not match n(={n})/m(={m})")

    data.num_tokens = N
    data.H = H
    data.adjs = [edge_index]
    data.x = x

    return data