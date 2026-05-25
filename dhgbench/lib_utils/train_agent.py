import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm,trange
from lib_utils.utils import mask_to_index,relabel_hyperedge_index,add_self_loop_hyperedges
from lib_utils.metrics import evaluate,evaluate_edge,edge_evaluation_printer,hg_evaluation_printer,evaluate_hypegraph
from lib_models import _semi_methods_
from sklearn.utils.class_weight import compute_class_weight
from lib_models.HNN.preprocessing import algo_preprocessing

class Trainer:
    
    def __init__(self, args, **kwargs):
        """
        Training pipline for different kinds of models
        """
        self.args = args
        self.device=args.device
    
    def semi_hg_cls_training(self,model,data,batch_loaders,args):

        device = args.device

        if args.use_weighted_loss:
            class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(data.y.numpy()), y=data.y.numpy())
            class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            class_weights = None  
        
        criterion = nn.NLLLoss(weight=class_weights.to(device) if args.use_weighted_loss else None)
        
        start_time = time.time()
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        train_loader, val_loader, test_loader = batch_loaders['train'], batch_loaders['val'], batch_loaders['test']
        
        best_val_acc = -1 
        best_model = None

        for epoch in tqdm(range(args.epochs)):
            
            # Training part
            model.train()
            optimizer.zero_grad()
            
            total_loss = 0
            
            for batch in train_loader:
                
                if args.method in ['HyperND','TFHNN']:
                    model.encoder.cache=None 
                elif args.method in ['HyperGCN']:
                    model.encoder.structure=None
                elif args.method in ['SheafHyperGNN']:
                    model.encoder.hyperedge_attr=None 
                    
                if args.method in ['HyperND','TFHNN','HyperGCN','SheafHyperGNN']:
                    reindex_hyperedge_index,_ = relabel_hyperedge_index(batch.hyperedge_index)
                    batch.hyperedge_index = reindex_hyperedge_index
                    add_loop_hyperedge_index = add_self_loop_hyperedges(batch.hyperedge_index, batch.num_nodes)
                    batch.hyperedge_index = add_loop_hyperedge_index 
                
                batch = algo_preprocessing(batch,args) 
                
                if args.method in ['AllSetformer']:
                    batch.norm = torch.ones_like(batch.hyperedge_index[0])

                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                out = F.log_softmax(out, dim=1)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader) 
            
            if (epoch+1) % args.display_step == 0:
                
                result = evaluate_hypegraph(model,batch_loaders,args) 
                
                if result['acc'][1]>=best_val_acc:
                    best_val_acc = result['acc'][1]
                    best_model = copy.deepcopy(model)

                print(f'Epoch: {epoch+1:02d}, Training loss: {avg_loss:.4f}')
                hg_evaluation_printer(result)
                
        end_time = time.time()
        print(f'Training Time: {end_time-start_time:.2f}')
        
        if args.early_stop:
            
            return best_model
        else:
            return model

    def semi_edge_pred_training(self,model,data,batch_loaders,args):

        start_time = time.time()
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        train_pos_loader = batch_loaders['train_pos']
        train_neg_loader = batch_loaders['train_neg']

        best_val_auc = -1 
        best_model = None

        for epoch in tqdm(range(args.epochs)):
            
            # Training part
            model.train()
            total_loss = 0.0
            
            while True :
                
                optimizer.zero_grad()
                
                n,_ = model.encoding(data) 
                
                pos_hedges, pos_labels, is_last = train_pos_loader.next()
                neg_hedges, neg_labels, is_last = train_neg_loader.next()

                pos_preds = model.aggregate(n, pos_hedges, mode='Train') 
                neg_preds = model.aggregate(n, neg_hedges, mode='Train') 
                
                d_real_loss = F.binary_cross_entropy_with_logits(pos_preds, pos_labels) 
                d_fake_loss = F.binary_cross_entropy_with_logits(neg_preds, neg_labels) 
                train_loss = d_real_loss + d_fake_loss
                
                train_loss.backward()
                optimizer.step()

                total_loss += train_loss.item()
                
                if is_last :
                    break 
            
            should_evaluate = (epoch+1) % args.display_step == 0 or (epoch+1) == args.epochs
            if should_evaluate:
                
                print(f'Epoch: {epoch+1:02d}, Training loss: {total_loss:.4f}')
                
                train_metrics, val_metrics, test_metrics = evaluate_edge(model,data,batch_loaders)

                if val_metrics['roc_average']>=best_val_auc:

                    best_val_auc = val_metrics['roc_average']
                    best_model = copy.deepcopy(model)

                edge_evaluation_printer(train_metrics, val_metrics, test_metrics)
        
        end_time = time.time()
        print(f'Training Time: {end_time-start_time:.2f}')

        if args.early_stop:
            if best_model is None:
                best_model = copy.deepcopy(model)
            return best_model
        else:
            return model
    
    def semi_node_cls_training(self,model,data,masks,args):
        
        criterion = nn.NLLLoss()
        
        model.train()
        
        ### Training loop ###
        start_time = time.time()
        model.reset_parameters()
        
        if args.method == 'UniGCNII':
            optimizer = torch.optim.Adam([
                dict(params=model.reg_params, weight_decay=0.01),
                dict(params=model.non_reg_params, weight_decay=5e-4)
            ], lr=0.01)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        if args.method == 'HJRL':
            pos_weight_H = float(data.H_ini.shape[0] * data.H_ini.shape[0] - data.H_ini.sum()) / data.H_ini.sum()
            if torch.isinf(pos_weight_H):
                pos_weight_H = torch.FloatTensor(args.pos_weight_thresh)
        elif args.method == 'TMPHN':
            train_idx = mask_to_index(masks['train'].to('cpu'),data.x.shape[0])
            data.target_idx = train_idx

        #for epoch in tqdm(range(args.epochs)):
        for epoch in trange(args.epochs):
        # for epoch in range(args.epochs):
            # Training part
            model.train()
            optimizer.zero_grad()
            if args.method == 'HJRL':
                node_embed,edge_embed = model(data)
                # 1. node classification loss
                node_embed = F.log_softmax(node_embed, dim=1)
                loss_1 = criterion(node_embed[masks['train']],data.y[masks['train']])
                if args.gamma == 0:
                    loss_2 = 0
                else:
                    # 2. reconstruction loss
                    recovered_H = torch.mm(node_embed, edge_embed.t())
                    recovered_H = torch.sigmoid(recovered_H)
                    if args.sample_ratio:
                        sampled_row = np.random.choice(recovered_H.shape[0], int(recovered_H.shape[0] * args.sample_ratio), replace=False)
                        loss_2 = F.binary_cross_entropy_with_logits(recovered_H[sampled_row], data.H_ini[sampled_row], pos_weight=pos_weight_H)
                    else:
                        loss_2 = F.binary_cross_entropy_with_logits(recovered_H, data.H_ini, pos_weight=pos_weight_H)
                loss = loss_1 + args.gamma * loss_2
                loss.backward()
                optimizer.step()
            else:
                out,_ = model(data) 
                out = F.log_softmax(out, dim=1)
                if args.method == 'TMPHN':
                    loss = criterion(out,data.y[masks['train']])
                else:
                    loss = criterion(out[masks['train']],data.y[masks['train']])
                loss.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_thresh)
                optimizer.step()

            if (epoch+1) % args.display_step == 0:
                
                if args.method == 'TMPHN':
                    data.target_idx=data.all_idx
                    
                result = evaluate(model, data, masks) 
                
                print(f'Epoch: {epoch+1:02d}, '
                    f'Train Acc: {100 * result[0]:.2f}%, '
                    f'Valid Acc: {100 * result[1]:.2f}%, '
                    f'Test  Acc: {100 * result[2]:.2f}%')

                if args.method == 'TMPHN':
                    data.target_idx=train_idx
                
        end_time = time.time()
        self.train_time = end_time-start_time
        print(f'Training Time: {end_time-start_time:.2f}')
        
        if args.method == 'TMPHN':
            data.target_idx=data.all_idx 
   
    def training(self,model,data,args,seed_split=None,task_type='node_cls'):
        
        if self.args.method in _semi_methods_:
            if task_type == 'node_cls':
                self.semi_node_cls_training(model,data,seed_split,args)
            elif task_type == 'edge_pred':
                model = self.semi_edge_pred_training(model,data,seed_split,args)
                return model
            elif task_type == 'hg_cls':
                model = self.semi_hg_cls_training(model,data,seed_split,args)
                return model
            else:
                raise NotImplementedError
