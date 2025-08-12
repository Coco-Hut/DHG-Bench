import os
import random
import torch
import numpy as np
from lib_utils.utils import fix_seed
from lib_dataset.edge_sampler import *

def generate_split_hyperedges(data,args,seed):

    hyperedge_index = data.hyperedge_index.to('cpu')

    hyperedges = defaultdict(set)

    for node, edge in zip(*hyperedge_index.tolist()):
        hyperedges[edge].add(node)

    HE = [frozenset(nodes) for nodes in hyperedges.values()]
    
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
    
    train_mns, train_sns, train_cns = neg_generator(GP_train, ground_train_num+train_only_num)
    valid_mns, valid_sns, valid_cns = neg_generator(GP_valid, ground_valid_num+valid_only_num)
    test_mns, test_sns, test_cns = neg_generator(GP_test, test_num)
    
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
        f'./lib_edge_splits/{args.dname}/split_{seed}.pt')

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
