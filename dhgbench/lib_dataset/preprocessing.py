import copy
import torch
import numpy as np
from collections import Counter
from torch_scatter import scatter_add
from torch_geometric.nn.conv.gcn_conv import gcn_norm

'''-----------------Hypernode/edge-level Task Datasets Preprocessing'''

def data_processing(args,db):
    
    data=copy.deepcopy(db.data)
    data = ExtractV2E(data)

    preserve_canonical_hyperedges = (
        getattr(args, 'task_type', None) == 'edge_pred'
        and getattr(args, 'edge_pred_protocol', 'legacy') == 'observed'
    )
    if preserve_canonical_hyperedges:
        # Preserve prediction targets before method-only graph augmentation.
        canonical_hyperedge_index = data.edge_index.detach().cpu().clone()
        canonical_hyperedge_index[1] -= canonical_hyperedge_index[1].min()
    
    if args.method in ['AllSetformer', 'AllDeepSets']:
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        if args.exclude_self:
            data = expand_edge_index(data)
        data = norm_contruction(data, option=args.normtype)
        db.norm=data.norm.to(args.device)
    else:
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        data.edge_index[1] -= data.edge_index[1].min()
    
    if db.sens is not None:
        db.sens=db.sens.to(args.device)
    
    db.x=data.x.to(args.device)
    if preserve_canonical_hyperedges:
        db.canonical_hyperedge_index=canonical_hyperedge_index
    elif hasattr(db, 'canonical_hyperedge_index'):
        del db.canonical_hyperedge_index
    db.hyperedge_index=data.edge_index.to(args.device)
    db.y=data.y.to(args.device)
    db.data=data
    
    return db

def expand_edge_index(data, edge_th=0):

    edge_index = data.edge_index
    num_nodes = data.n_x[0].item()
    if hasattr(data, 'totedges'):
        num_edges = data.totedges
    else:
        num_edges = data.num_hyperedges[0]

    expanded_n2he_index = []

    cur_he_id = num_nodes

    new_edge_id_2_original_edge_id = {}

    for he_idx in range(num_nodes, num_edges + num_nodes):
        # find all nodes within the same hyperedge.
        selected_he = edge_index[:, edge_index[1] == he_idx]
        size_of_he = selected_he.shape[1]

#         Trim a hyperedge if its size>edge_th
        if edge_th > 0:
            if size_of_he > edge_th:
                continue

        if size_of_he == 1:

            new_n2he = selected_he.clone()
            new_n2he[1] = cur_he_id
            expanded_n2he_index.append(new_n2he)

            cur_he_id += 1
            continue

        new_n2he = selected_he.repeat_interleave(size_of_he, dim=1)

        # new_edge_ids start from the he_id from previous iteration (cur_he_id).
        new_edge_ids = torch.LongTensor(
            np.arange(cur_he_id, cur_he_id + size_of_he)).repeat(size_of_he)
        new_n2he[1] = new_edge_ids

        # build a mapping between node and it's corresponding edge.
        # e.g. {n_1: e_7_1, n_2: e_7_2}
        tmp_node_id_2_he_id_dict = {}
        for idx in range(size_of_he):
            new_edge_id_2_original_edge_id[cur_he_id] = he_idx
            cur_node_id = selected_he[0][idx].item()
            tmp_node_id_2_he_id_dict[cur_node_id] = cur_he_id
            cur_he_id += 1

        # create n2he by deleting the self-product edge.
        new_he_select_mask = torch.BoolTensor([True] * new_n2he.shape[1])
        for col_idx in range(new_n2he.shape[1]):
            tmp_node_id, tmp_edge_id = new_n2he[0, col_idx].item(
            ), new_n2he[1, col_idx].item()
            if tmp_node_id_2_he_id_dict[tmp_node_id] == tmp_edge_id:
                new_he_select_mask[col_idx] = False
        new_n2he = new_n2he[:, new_he_select_mask]
        expanded_n2he_index.append(new_n2he)

    new_edge_index = torch.cat(expanded_n2he_index, dim=1)

    new_order = new_edge_index[0].argsort()
    data.edge_index = new_edge_index[:, new_order]

    return data

def ExtractV2E(data):
    # Assume edge_index = [V|E;E|V]
    edge_index = data.edge_index
#     First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    num_nodes = data.n_x
    num_hyperedges = data.num_hyperedges
    #if not ((data.n_x[0]+data.num_hyperedges[0]-1) == data.edge_index[0].max().item()):
    if not ((data.n_x+data.num_hyperedges-1) == data.edge_index[0].max().item()):
        print('num_hyperedges does not match! 1')
        return
    cidx = torch.where(edge_index[0] == num_nodes)[
        0].min()  # cidx: [V...|cidx E...]
    data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)
    return data

def Add_Self_Loops(data):
    
    # update so we dont jump on some indices
    # Assume edge_index = [V;E]. If not, use ExtractV2E()
    edge_index = data.edge_index
    num_nodes = data.n_x
    num_hyperedges = data.num_hyperedges

    if not ((data.n_x + data.num_hyperedges - 1) == data.edge_index[1].max().item()):
        print('num_hyperedges does not match! 2')
        return

    hyperedge_appear_fre = Counter(edge_index[1].numpy())
    # store the nodes that already have self-loops
    skip_node_lst = []
    for edge in hyperedge_appear_fre:
        if hyperedge_appear_fre[edge] == 1:
            skip_node = edge_index[0][torch.where(
                edge_index[1] == edge)[0].item()]
            skip_node_lst.append(skip_node.item())

    new_edge_idx = edge_index[1].max() + 1
    new_edges = torch.zeros(
        (2, num_nodes - len(set(skip_node_lst))), dtype=edge_index.dtype) 
    tmp_count = 0
    for i in range(num_nodes):
        if i not in skip_node_lst:
            new_edges[0][tmp_count] = i
            new_edges[1][tmp_count] = new_edge_idx
            new_edge_idx += 1
            tmp_count += 1

    data.totedges = num_hyperedges + num_nodes - len(set(skip_node_lst)) 
    edge_index = torch.cat((edge_index, new_edges), dim=1)
    # Sort along w.r.t. nodes
    _, sorted_idx = torch.sort(edge_index[0])
    data.edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    return data

def norm_contruction(data, option='all_one', TYPE='V2E'):
    if TYPE == 'V2E':
        if option == 'all_one':
            data.norm = torch.ones_like(data.edge_index[0])

        elif option == 'deg_half_sym':
            edge_weight = torch.ones_like(data.edge_index[0])
            cidx = data.edge_index[1].min()
            Vdeg = scatter_add(edge_weight, data.edge_index[0], dim=0)
            HEdeg = scatter_add(edge_weight, data.edge_index[1]-cidx, dim=0)
            V_norm = Vdeg**(-1/2)
            E_norm = HEdeg**(-1/2)
            data.norm = V_norm[data.edge_index[0]] * \
                E_norm[data.edge_index[1]-cidx]

    elif TYPE == 'V2V':
        data.edge_index, data.norm = gcn_norm(
            data.edge_index, data.norm, add_self_loops=True)
    return data

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=False):
    """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    if not balance:
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]

        if not ignore_negative:
            return train_indices, val_indices, test_indices

        train_idx = labeled_nodes[train_indices]
        valid_idx = labeled_nodes[val_indices]
        test_idx = labeled_nodes[test_indices]

        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    else:
        #         ipdb.set_trace()
        indices = []
        for i in range(label.max()+1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop/(label.max()+1)*len(label))
        val_lb = int(valid_prop*len(label))
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    return split_idx

'''-----------------Hypergraph-level Task Datasets Preprocessing'''

def convert_to_hyperedge_index(e_list):
    node_list = []
    edge_list = []
    for edge_idx, cur_he in enumerate(e_list):
        cur_size = len(cur_he)

        node_list += cur_he 
        edge_list += [edge_idx] * cur_size

    edge_index = np.array([node_list,edge_list], dtype = int)
    edge_index = torch.LongTensor(edge_index)
    return edge_index

def compute_degree_list(hyperedge_index):
    node_ids = hyperedge_index[0]  
    max_node_id = node_ids.max().item()  
    degrees = torch.bincount(node_ids, minlength=max_node_id + 1) 
    return degrees.tolist()  
