import copy
import torch
from torch import Tensor

def perturbation(data,mode,p,masks=None):

    perturbed_data = copy.deepcopy(data)
    
    if mode == 'add_incidence':
        pt_hyperedge_index = add_incidence(data.hyperedge_index,p)
        perturbed_data.hyperedge_index = pt_hyperedge_index
    elif mode == 'drop_incidence':
        pt_hyperedge_index = drop_incidence(data.hyperedge_index,p,data.num_nodes,data.num_hyperedges)
        perturbed_data.hyperedge_index = pt_hyperedge_index
    elif mode == 'spar_feat':
        pt_feat = drop_features(data.x,p)
        perturbed_data.x = pt_feat
    elif mode == 'noise_feat':
        pt_feat = add_gaussian_noise(data.x,p)
        perturbed_data.x = pt_feat
    elif mode == 'spar_label':
        new_train_mask = mask_label(masks['train'], p=p)
        masks['train'] = new_train_mask
        return masks 
    elif mode == 'flip_label':
        perturbed_labels = perturb_train_labels(data.y,masks['train'],data.num_classes,p)
        perturbed_data.y = perturbed_labels
    else:
        raise NotImplementedError('Not Implemneted Perturbation')
    
    return perturbed_data

def drop_features(x: Tensor, p: float):

    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def add_gaussian_noise(x: Tensor, lam: float):

    r = x.max(dim=1, keepdim=True).values.mean()
    noise = torch.randn_like(x) * lam * r

    return x+noise

def filter_incidence(row: Tensor, col: Tensor, hyperedge_attr, mask: Tensor):
    return row[mask], col[mask], None if hyperedge_attr is None else hyperedge_attr[mask]

def drop_incidence(hyperedge_index: Tensor, p: float = 0.2, num_nodes: int = 0, num_edges: int = 0) -> Tensor:
    if p == 0.0:
        return hyperedge_index
    
    row, col = hyperedge_index
    mask = torch.rand(row.size(0), device=hyperedge_index.device) >= p
    
    row, col, _ = filter_incidence(row, col, None, mask)
    hyperedge_index = torch.stack([row, col], dim=0)

    # padding operation
    padding_edge = torch.tensor([[num_nodes-1], [num_edges-1]],device=hyperedge_index.device)
    hyperedge_index = torch.cat([hyperedge_index, padding_edge], dim=1)

    return hyperedge_index

def add_incidence(hyperedge_index: Tensor, p: float = 0.2):

    row, col = hyperedge_index
    num_nodes, num_edges= row.max()+1,col.max()+1

    num_add = int(row.size(0) * p) 
    
    if num_add > 0:
        existing_edges = set(zip(row.tolist(), col.tolist()))
        new_edges = set()
        
        while len(new_edges) < num_add:
            new_edge = (torch.randint(0, num_nodes, (1,)).item(), 
                        torch.randint(0, num_edges, (1,)).item())
            if new_edge not in existing_edges:
                new_edges.add(new_edge)
        
        new_rows, new_cols = zip(*new_edges)
        row = torch.cat([row, torch.tensor(new_rows, device=row.device)], dim=0)
        col = torch.cat([col, torch.tensor(new_cols, device=col.device)], dim=0)
    
    return torch.stack([row, col], dim=0)

def perturb_train_labels(labels: torch.Tensor, train_mask: torch.Tensor, num_classes: int, p: float = 0.1):

    perturbed_labels = labels.clone()
    
    train_indices = torch.where(train_mask)[0]
    num_perturb = int(len(train_indices) * p) 
    
    if num_perturb > 0:
        perturb_indices = train_indices[torch.randperm(len(train_indices))[:num_perturb]]
        
        random_labels = torch.randint(0, num_classes, (num_perturb,), device=labels.device)
        mask_diff = random_labels == labels[perturb_indices]
        while mask_diff.any(): 
            random_labels[mask_diff] = torch.randint(0, num_classes, (mask_diff.sum(),), device=labels.device)
            mask_diff = random_labels == labels[perturb_indices]
        
        perturbed_labels[perturb_indices] = random_labels
    
    return perturbed_labels

def mask_label(mask: torch.Tensor, p: float):

    assert mask.dtype == torch.bool

    true_indices = torch.nonzero(mask, as_tuple=False).squeeze()

    num_to_flip = int(len(true_indices) * p)

    if num_to_flip == 0:
        return mask.clone()  

    perm = torch.randperm(len(true_indices), device=mask.device)
    flip_indices = true_indices[perm[:num_to_flip]]

    new_mask = mask.clone()
    new_mask[flip_indices] = False

    return new_mask