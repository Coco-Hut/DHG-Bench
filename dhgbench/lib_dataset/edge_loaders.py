import os
import random
import copy
import torch
import numpy as np
from lib_utils.utils import fix_seed
from lib_dataset.edge_sampler import *

import torch
import numpy as np
from collections import defaultdict

def hyperedges_from_data(data):
    hyperedge_index = data.hyperedge_index.to('cpu')
    hyperedges = defaultdict(set)

    for node, edge in zip(*hyperedge_index.tolist()):
        hyperedges[edge].add(node)

    return deduplicate_hyperedges(hyperedges.values())

def deduplicate_hyperedges(hyperedges):
    unique_hyperedges = []
    seen = set()

    for edge in hyperedges:
        canonical_edge = frozenset(int(node) for node in edge)
        if canonical_edge in seen:
            continue
        seen.add(canonical_edge)
        unique_hyperedges.append(canonical_edge)

    return unique_hyperedges

def hyperedges_to_index(hyperedges, device):
    node_ids = []
    edge_ids = []
    for edge_id, edge in enumerate(hyperedges):
        for node in edge:
            node_ids.append(int(node))
            edge_ids.append(edge_id)
    if len(node_ids) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor([node_ids, edge_ids], dtype=torch.long, device=device)

def build_observed_support_data(data, data_dict, args):
    support_data = copy.deepcopy(data)
    support_hyperedges = [frozenset(edge) for edge in data_dict.get("support_pos", [])]
    support_hyperedge_index = hyperedges_to_index(support_hyperedges, args.device)

    support_data.hyperedge_index = support_hyperedge_index
    support_data.num_hyperedges = len(support_hyperedges)
    support_data.norm = torch.ones_like(support_hyperedge_index[0])

    if hasattr(support_data, "data"):
        support_data.data.edge_index = support_hyperedge_index.detach().cpu()
        support_data.data.num_hyperedges = torch.tensor([len(support_hyperedges)])
        support_data.data.norm = torch.ones_like(support_data.data.edge_index[0])

    return support_data

def generate_ind_split_hyperedges(data, args, seed):
    """
    hyperedges: list[frozenset]
    ratio: train/val/test node ratio
    """

    hyperedges = hyperedges_from_data(data)

    ratio = (args.train_prop, args.valid_prop, 1 - args.train_prop - args.valid_prop)

    seed_base = 42  
    fix_seed(seed_base+seed) 

    all_nodes = set()
    for e in hyperedges:
        all_nodes |= e
    all_nodes = list(all_nodes)
    np.random.shuffle(all_nodes)

    N = len(all_nodes)
    r1, r2, r3 = ratio

    T1 = int(N * r1)
    T2 = int(N * (r1 + r2))

    train_nodes = set(all_nodes[:T1])
    valid_nodes = set(all_nodes[T1:T2])
    test_nodes  = set(all_nodes[T2:])

    GP_train = []
    GP_valid = []
    GP_test  = []

    for e in hyperedges:
        if e.issubset(train_nodes):
            GP_train.append(e)
        elif e.issubset(valid_nodes):
            GP_valid.append(e)
        elif e.issubset(test_nodes):
            GP_test.append(e)

    train_size, valid_size, test_size = len(GP_train), len(GP_valid), len(GP_test)

    print(f'train_size: {train_size}, valid_size: {valid_size}, test_size: {test_size}')

    all_positive_hyperedges = set(hyperedges)
    train_mns, train_sns, train_cns = neg_generator_excluding(GP_train, train_size, all_positive_hyperedges)
    valid_mns, valid_sns, valid_cns = neg_generator_excluding(GP_valid, valid_size, all_positive_hyperedges)
    test_mns, test_sns, test_cns = neg_generator_excluding(GP_test, test_size, all_positive_hyperedges)

    # positive samples
    ground_train_data = []
    ground_valid_data = []
    train_only_data = [list(edge) for edge in GP_train]
    valid_only_data = [list(edge) for edge in GP_valid]
    test_data = [list(edge) for edge in GP_test]

    torch.save({'ground_train': ground_train_data, 'ground_valid': ground_valid_data, \
    'train_only_pos': train_only_data, 'train_mns': train_mns, 'train_sns' : train_sns, 'train_cns' : train_cns,\
    'valid_only_pos': valid_only_data, 'valid_mns': valid_mns, 'valid_sns' : valid_sns, 'valid_cns' : valid_cns, \
    'test_pos': test_data, 'test_mns': test_mns, 'test_sns' : test_sns, 'test_cns' : test_cns},
    f'./lib_edge_splits/{args.edge_split_mode}/{args.dname}/split_{seed}.pt')

def generate_observed_ind_split_hyperedges(data, args, seed):
    raise ValueError(
        "The observed edge prediction protocol currently supports only "
        "--edge_split_mode=trand. The observed ind split is disabled because "
        "its support graph contains the train positives."
    )

    HE = hyperedges_from_data(data)

    ratio = (args.train_prop, args.valid_prop, 1 - args.train_prop - args.valid_prop)

    seed_base = 42
    fix_seed(seed_base+seed)

    all_nodes = set()
    for e in HE:
        all_nodes |= e
    all_nodes = list(all_nodes)
    np.random.shuffle(all_nodes)

    N = len(all_nodes)
    r1, r2, r3 = ratio

    T1 = int(N * r1)
    T2 = int(N * (r1 + r2))

    train_nodes = set(all_nodes[:T1])
    valid_nodes = set(all_nodes[T1:T2])
    test_nodes  = set(all_nodes[T2:])

    GP_train = []
    GP_valid = []
    GP_test  = []

    for e in HE:
        if e.issubset(train_nodes):
            GP_train.append(e)
        elif e.issubset(valid_nodes):
            GP_valid.append(e)
        elif e.issubset(test_nodes):
            GP_test.append(e)

    train_size, valid_size, test_size = len(GP_train), len(GP_valid), len(GP_test)

    print(f'train_size: {train_size}, valid_size: {valid_size}, test_size: {test_size}')

    all_positive_hyperedges = set(HE)
    train_mns, train_sns, train_cns = neg_generator_excluding(GP_train, train_size, all_positive_hyperedges)
    valid_mns, valid_sns, valid_cns = neg_generator_excluding(GP_train + GP_valid, valid_size, all_positive_hyperedges)
    test_mns, test_sns, test_cns = neg_generator_excluding(GP_train + GP_valid, test_size, all_positive_hyperedges)

    ground_train_data = []
    ground_valid_data = []
    train_only_data = [list(edge) for edge in GP_train]
    valid_only_data = [list(edge) for edge in GP_valid]
    test_data = [list(edge) for edge in GP_test]
    support_data = [list(edge) for edge in GP_train]

    torch.save({'edge_pred_protocol': 'observed', 'support_pos': support_data,
    'ground_train': ground_train_data, 'ground_valid': ground_valid_data, \
    'train_only_pos': train_only_data, 'train_mns': train_mns, 'train_sns' : train_sns, 'train_cns' : train_cns,\
    'valid_only_pos': valid_only_data, 'valid_mns': valid_mns, 'valid_sns' : valid_sns, 'valid_cns' : valid_cns, \
    'test_pos': test_data, 'test_mns': test_mns, 'test_sns' : test_sns, 'test_cns' : test_cns},
    f'{args.edge_save_dir}{args.edge_split_mode}_{args.edge_pred_protocol}/{args.dname}/split_{seed}.pt')

def generate_split_hyperedges(data,args,seed):

    HE = hyperedges_from_data(data)
    
    base_cover = get_cover_idx(HE)
    union = get_union(HE)
    tmp = [HE[idx] for idx in base_cover]
    assert union == get_union(tmp)
    base_num = len(base_cover)
    
    os.makedirs(args.edge_save_dir+args.dname, exist_ok=True)  
    
    seed_base = 42  
    fix_seed(seed_base+seed) 
    
    # ground 60%, train 10(+50)%, validation 10(+10)%, test 20%
    # ground_num = int(0.6*len(HE)) - base_num
    ground_num = max(int(0.6*len(HE)) - base_num,0)
    total_idx = list(range(len(HE))) 
    ground_idx = list(set(total_idx)-set(base_cover))
    ground_idx = random.sample(ground_idx, ground_num)      
    ground_num += base_num
    ground_idx += base_cover
    ground_valid_num = ground_num//6
    ground_valid_idx = random.sample(ground_idx, ground_valid_num)
    ground_train_num = ground_num - ground_valid_num
    
    ground_train_data = []
    ground_valid_data = []
    pred_data = []
    for idx in total_idx :
        if idx in ground_idx:
            if idx in ground_valid_idx:
                ground_valid_data.append(HE[idx])
            else:
                ground_train_data.append(HE[idx])
        else :
            pred_data.append(HE[idx])
            
    valid_only_num = int(0.25*len(pred_data))
    train_only_num = int(0.25*len(pred_data))
    test_num = len(pred_data) - (valid_only_num + train_only_num)
    
    random.shuffle(pred_data)
    train_only_data = pred_data[:train_only_num]
    valid_only_data = pred_data[train_only_num:-test_num]
    test_data = pred_data[-test_num:]
    
    # negative sampling        
    GP_train = ground_valid_data + ground_train_data + train_only_data
    GP_valid = ground_valid_data + ground_train_data + train_only_data + valid_only_data
    GP_test = GP_valid
    
    all_positive_hyperedges = set(HE)
    train_mns, train_sns, train_cns = neg_generator_excluding(GP_train, ground_train_num+train_only_num, all_positive_hyperedges)
    valid_mns, valid_sns, valid_cns = neg_generator_excluding(GP_valid, ground_valid_num+valid_only_num, all_positive_hyperedges)
    test_mns, test_sns, test_cns = neg_generator_excluding(GP_test, test_num, all_positive_hyperedges)
    
    # positive samples
    ground_train_data = [list(edge) for edge in ground_train_data]
    ground_valid_data = [list(edge) for edge in ground_valid_data]
    train_only_data = [list(edge) for edge in train_only_data]
    valid_only_data = [list(edge) for edge in valid_only_data]
    test_data = [list(edge) for edge in test_data]
    
    print(f"ground {len(ground_train_data)} + {len(ground_valid_data)} = {len(ground_train_data + ground_valid_data)}")
    print(f"train pos {len(ground_train_data)} + {len(train_only_data)} = {len(ground_train_data + train_only_data)}, neg {len(train_mns)}")
    print(f"valid pos {len(ground_valid_data)} + {len(valid_only_data)} = {len(ground_valid_data + valid_only_data)}, neg {len(valid_mns)}")
    print(f"test pos {len(test_data)}, neg {len(test_mns)}")
    
    torch.save({'ground_train': ground_train_data, 'ground_valid': ground_valid_data, \
        'train_only_pos': train_only_data, 'train_mns': train_mns, 'train_sns' : train_sns, 'train_cns' : train_cns,\
        'valid_only_pos': valid_only_data, 'valid_mns': valid_mns, 'valid_sns' : valid_sns, 'valid_cns' : valid_cns, \
        'test_pos': test_data, 'test_mns': test_mns, 'test_sns' : test_sns, 'test_cns' : test_cns},
        f'./lib_edge_splits/{args.edge_split_mode}/{args.dname}/split_{seed}.pt')

def generate_observed_split_hyperedges(data,args,seed):

    HE = hyperedges_from_data(data)

    base_cover = get_cover_idx(HE)
    union = get_union(HE)
    tmp = [HE[idx] for idx in base_cover]
    assert union == get_union(tmp)
    base_num = len(base_cover)

    os.makedirs(f'{args.edge_save_dir}{args.edge_split_mode}_{args.edge_pred_protocol}/{args.dname}', exist_ok=True)

    seed_base = 42
    fix_seed(seed_base+seed)

    ground_num = max(int(0.6*len(HE)) - base_num,0)
    total_idx = list(range(len(HE)))
    ground_idx = list(set(total_idx)-set(base_cover))
    ground_idx = random.sample(ground_idx, ground_num)
    ground_num += base_num
    ground_idx += base_cover
    ground_valid_num = ground_num//6
    ground_valid_idx = random.sample(ground_idx, ground_valid_num)
    ground_train_num = ground_num - ground_valid_num

    ground_train_data = []
    ground_valid_data = []
    pred_data = []
    for idx in total_idx :
        if idx in ground_idx:
            if idx in ground_valid_idx:
                ground_valid_data.append(HE[idx])
            else:
                ground_train_data.append(HE[idx])
        else :
            pred_data.append(HE[idx])

    valid_only_num = int(0.25*len(pred_data))
    train_only_num = int(0.25*len(pred_data))
    test_num = len(pred_data) - (valid_only_num + train_only_num)

    random.shuffle(pred_data)
    train_only_data = pred_data[:train_only_num]
    valid_only_data = pred_data[train_only_num:-test_num]
    test_data = pred_data[-test_num:]

    all_positive_hyperedges = set(HE)
    GP_train = ground_train_data + train_only_data
    GP_valid = ground_train_data + train_only_data + ground_valid_data + valid_only_data
    GP_test = GP_valid

    train_mns, train_sns, train_cns = neg_generator_excluding(GP_train, train_only_num, all_positive_hyperedges)
    valid_mns, valid_sns, valid_cns = neg_generator_excluding(GP_valid, ground_valid_num+valid_only_num, all_positive_hyperedges)
    test_mns, test_sns, test_cns = neg_generator_excluding(GP_test, test_num, all_positive_hyperedges)

    support_data = [list(edge) for edge in ground_train_data]
    ground_train_data = [list(edge) for edge in ground_train_data]
    ground_valid_data = [list(edge) for edge in ground_valid_data]
    train_only_data = [list(edge) for edge in train_only_data]
    valid_only_data = [list(edge) for edge in valid_only_data]
    test_data = [list(edge) for edge in test_data]

    print(f"support {len(support_data)}")
    print(f"train pos {len(train_only_data)}, neg {len(train_mns)}")
    print(f"valid pos {len(ground_valid_data)} + {len(valid_only_data)} = {len(ground_valid_data + valid_only_data)}, neg {len(valid_mns)}")
    print(f"test pos {len(test_data)}, neg {len(test_mns)}")

    torch.save({'edge_pred_protocol': 'observed', 'support_pos': support_data,
        'ground_train': ground_train_data, 'ground_valid': ground_valid_data, \
        'train_only_pos': train_only_data, 'train_mns': train_mns, 'train_sns' : train_sns, 'train_cns' : train_cns,\
        'valid_only_pos': valid_only_data, 'valid_mns': valid_mns, 'valid_sns' : valid_sns, 'valid_cns' : valid_cns, \
        'test_pos': test_data, 'test_mns': test_mns, 'test_sns' : test_sns, 'test_cns' : test_cns},
        f'{args.edge_save_dir}{args.edge_split_mode}_{args.edge_pred_protocol}/{args.dname}/split_{seed}.pt')

def generate_edge_loaders(data_dict, args):

    device = args.device
    
    train_pos_loader = load_train(data_dict, args.edge_batch_size, device,label="pos") # only positives
    train_neg_loader = load_train(data_dict, args.edge_batch_size, device,label=args.ns_method) # only positives

    val_pos_loader = load_val(data_dict, args.edge_batch_size, device, label="pos")
    val_neg_sns_loader = load_val(data_dict, args.edge_batch_size, device, label="sns")
    val_neg_mns_loader = load_val(data_dict, args.edge_batch_size, device, label="mns")
    val_neg_cns_loader = load_val(data_dict, args.edge_batch_size, device, label="cns")

    test_pos_loader = load_test(data_dict, args.edge_batch_size, device, label="pos")
    test_neg_sns_loader = load_test(data_dict, args.edge_batch_size, device, label="sns")
    test_neg_mns_loader = load_test(data_dict, args.edge_batch_size, device, label="mns")
    test_neg_cns_loader = load_test(data_dict, args.edge_batch_size, device, label="cns")

    batch_loaders = {
        'train_pos': train_pos_loader,
        'train_neg': train_neg_loader,
        'val_pos': val_pos_loader,
        'val_neg_sns':val_neg_sns_loader,
        'val_neg_mns':val_neg_mns_loader,
        'val_neg_cns':val_neg_cns_loader,
        'test_pos': test_pos_loader,
        'test_neg_sns':test_neg_sns_loader,
        'test_neg_mns':test_neg_mns_loader,
        'test_neg_cns':test_neg_cns_loader,
    }
    
    return batch_loaders

def load_train(data_dict, bs, device, label):
    if label=="pos":
        if data_dict.get("edge_pred_protocol") == "observed":
            train_pos = data_dict["train_only_pos"]
        else:
            train_pos = data_dict["train_only_pos"] + data_dict["ground_train"]
        train_pos_label = [1 for i in range(len(train_pos))]
        train_batchloader = HEBatchGenerator(train_pos, train_pos_label, bs, device, test_generator=False) 
    elif label =="mixed":
        d = len(data_dict["train_sns"]) // 3
        train_neg = data_dict["train_sns"][0:d] + data_dict["train_mns"][0:d] + data_dict["train_cns"][0:d]
        train_neg_label = [0 for i in range(len(train_neg))]
        train_batchloader = HEBatchGenerator(train_neg, train_neg_label, bs, device, test_generator=False) 
    else:
        train_neg = data_dict[f"train_{label}"]
        train_neg_label = [0 for i in range(len(train_neg))]
        train_batchloader = HEBatchGenerator(train_neg, train_neg_label, bs, device, test_generator=False) 
    
    return train_batchloader

def load_val(data_dict, bs, device, label):
    if label=="pos":
        if data_dict.get("edge_pred_protocol") == "observed":
            val = data_dict["ground_valid"] + data_dict["valid_only_pos"]
        else:
            val = data_dict["train_only_pos"] + data_dict["ground_train"]
        val_label = [1 for i in range(len(val))]
    else:
        val = data_dict[f"valid_{label}"]
        val_label = [0 for i in range(len(val))]
    val_batchloader = HEBatchGenerator(val, val_label, bs, device, test_generator=True)    
    return val_batchloader

def load_test(data_dict, bs, device, label):
    test = data_dict[f"test_{label}"]
    if label=="pos":
        test_label = [1 for i in range(len(test))]
    else:
        test_label = [0 for i in range(len(test))]
    test_batchloader = HEBatchGenerator(test, test_label, bs, device, test_generator=True)    
    return test_batchloader

# Batch Generator
class HEBatchGenerator(object):
    def __init__(self, hyperedges, labels, batch_size, device, test_generator=False):
        """Creates an instance of HyperedgeGroupBatchGenerator.
        
        Args:
            hyperedges: List(frozenset). List of hyperedges.
            labels: list. Labels of hyperedges.
            batch_size. int. Batch size of each batch.
            test_generator: bool. Whether batch generator is test generator.
        """
        self.batch_size = batch_size
        self.hyperedges = hyperedges
        self.labels = labels
        self._cursor = 0
        self.device = device
        self.test_generator = test_generator
        self.shuffle()
    
    def eval(self):
        self.test_generator = True

    def train(self):
        self.test_generator = False

    def shuffle(self):
        idcs = np.arange(len(self.hyperedges))
        np.random.shuffle(idcs)
        self.hyperedges = [self.hyperedges[i] for i in idcs]
        self.labels = [self.labels[i] for i in idcs]
  
    def __iter__(self):
        self._cursor = 0
        return self
    
    def next(self):
        return self.__next__()

    def __next__(self):
        if self.test_generator:
            return self.next_test_batch()
        else:
            return self.next_train_batch()

    def next_train_batch(self):
        ncursor = self._cursor+self.batch_size
        if ncursor >= len(self.hyperedges):
            hyperedges = self.hyperedges[self._cursor:] + self.hyperedges[
                :ncursor - len(self.hyperedges)]

            labels = self.labels[self._cursor:] + self.labels[
                :ncursor - len(self.labels)]
          
            self._cursor = ncursor - len(self.hyperedges)
            hyperedges = [torch.LongTensor(edge).to(self.device) for edge in hyperedges]
            labels = torch.FloatTensor(labels).to(self.device)
            self.shuffle()
            return hyperedges, labels, True
        
        hyperedges = self.hyperedges[
            self._cursor:self._cursor + self.batch_size]
        
        labels = self.labels[
            self._cursor:self._cursor + self.batch_size]
        
        hyperedges = [torch.LongTensor(edge).to(self.device) for edge in hyperedges]
        labels = torch.FloatTensor(labels).to(self.device)
       
        self._cursor = ncursor % len(self.hyperedges)
        return hyperedges, labels, False

    def next_test_batch(self):
        ncursor = self._cursor+self.batch_size
        if ncursor >= len(self.hyperedges):
            hyperedges = self.hyperedges[self._cursor:]
            labels = self.labels[self._cursor:]
            self._cursor = 0
            hyperedges = [torch.LongTensor(edge).to(self.device) for edge in hyperedges]
            labels = torch.FloatTensor(labels).to(self.device)
            
            return hyperedges, labels, True
        
        hyperedges = self.hyperedges[
            self._cursor:self._cursor + self.batch_size]
        
        labels = self.labels[
            self._cursor:self._cursor + self.batch_size]

        hyperedges = [torch.LongTensor(edge).to(self.device) for edge in hyperedges]
        labels = torch.FloatTensor(labels).to(self.device)
       
        self._cursor = ncursor % len(self.hyperedges)
        return hyperedges, labels, False

def get_union(union):
    ind = []
    for s in union :
        ind+=list(s)
    return set(ind)

def set_cover(universe, subsets):
    elements = set(e for s in subsets for e in s)
    if elements != universe:
        return None, None
    covered = set()
    cover = []
    idx = []
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s - covered))
        cover.append(subset)
        idx.append(subsets.index(subset))
        covered |= subset
    return cover, idx

def get_cover_idx(HE):
    universe = get_union(HE)
    tmp_HE = [set(edge) for edge in HE]
    _, cover_idx = set_cover(universe, tmp_HE)
    return cover_idx
