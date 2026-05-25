import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from sklearn.metrics import f1_score, roc_auc_score,average_precision_score,accuracy_score
from collections import defaultdict
from lib_models.HNN.preprocessing import algo_preprocessing
from lib_utils.utils import relabel_hyperedge_index,add_self_loop_hyperedges

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

#     ipdb.set_trace()
#     for i in range(y_true.shape[1]):
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

@torch.no_grad()
def evaluate(model, data, split_idx, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out,_ = model(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_acc(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_acc(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_acc(
        data.y[split_idx['test']], out[split_idx['test']])
    
    return train_acc, valid_acc, test_acc

def evaluate_hypegraph(model,batch_loaders,args):
    metrics_dict = defaultdict(list)
    for key,data_loaders in batch_loaders.items():
        acc,macro_f1 = test_hypegraph_loader(model,data_loaders,args)
        metrics_dict['acc'].append(acc)
        metrics_dict['macro_f1'].append(macro_f1)
    return metrics_dict 

def test_hypegraph_loader(model,data_loader,args):
    
    device = args.device
    
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():

        for batch in data_loader:
            
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
            out = model(batch)
            out = F.log_softmax(out, dim=1)  
            pred = out.argmax(dim=1)  
            
            all_preds.append(pred.cpu())  
            all_labels.append(batch.y.cpu())  
    
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(y_true, y_pred, average='macro')

    return accuracy, macro_f1

def hg_evaluation_printer(result):
    print(f'Train Acc: {100 * result["acc"][0]:.2f}%, '
        f'Valid Acc: {100 * result["acc"][1]:.2f}%, '
        f'Test  Acc: {100 * result["acc"][2]:.2f}%')
    
    print(f'Train Macro-F1: {100 * result["macro_f1"][0]:.2f}%, '
        f'Valid Macro-F1: {100 * result["macro_f1"][1]:.2f}%, '
        f'Test  Macro-F1: {100 * result["macro_f1"][2]:.2f}%')

def evaluate_edge(model,data,batch_loaders):
    
    train_metrics = eval_edge_train(model,data,batch_loaders)
    
    val_metrics = eval_edge_val_test(model,data,batch_loaders,mode='val')
    
    test_metrics = eval_edge_val_test(model,data,batch_loaders,mode='test')
    
    return train_metrics, val_metrics, test_metrics

def edge_evaluation_printer(train_metrics, val_metrics, test_metrics):

    print('---------Train set-----------')
    print(f"|AUROC| {train_metrics['roc_train']:.4f}")
    print(f"|AP| {train_metrics['ap_train']:.4f}")

    print('---------Valid set-----------')
    print(f"|AUROC| SNS: {val_metrics['roc_sns']:.4f} | MNS:{val_metrics['roc_mns']:.4f} | CNS: {val_metrics['roc_cns']:.4f} | MIX: {val_metrics['roc_mixed']:.4f} | AVG: {val_metrics['roc_average']:.4f}")
    print(f"|AP| SNS: {val_metrics['ap_sns']:.4f} | MNS:{val_metrics['ap_mns']:.4f} | CNS: {val_metrics['ap_cns']:.4f} | MIX: {val_metrics['ap_mixed']:.4f} | AVG: {val_metrics['ap_average']:.4f}")
    
    print('---------Test set-----------')
    print(f"|AUROC| SNS: {test_metrics['roc_sns']:.4f} | MNS:{test_metrics['roc_mns']:.4f} | CNS: {test_metrics['roc_cns']:.4f} | MIX: {test_metrics['roc_mixed']:.4f} | AVG: {test_metrics['roc_average']:.4f}")
    print(f"|AP| SNS: {test_metrics['ap_sns']:.4f} | MNS:{test_metrics['ap_mns']:.4f} | CNS: {test_metrics['ap_cns']:.4f} | MIX: {test_metrics['ap_mixed']:.4f} | AVG: {test_metrics['ap_average']:.4f}")

def aggr_metrics(metrics_dict,result):

    for data_type in result.keys():
        for k,v in result[data_type].items():
            metrics_dict[data_type][k].append(v)
    
    return metrics_dict

def avg_result_printer_edge(metrics_dict):
    
    print('---------Train set-----------')
    train_mean,train_std=[],[]
    for k in metrics_dict['train']:
        train_mean.append(np.mean(metrics_dict['train'][k]))
        train_std.append(np.std(metrics_dict['train'][k]))
    print(f"|AUROC| {train_mean[0]:.4f}+-{train_std[0]:.4f}")
    print(f"|AP| {train_mean[1]:.4f}+-{train_std[1]:.4f}")

    print('---------Valid set-----------')
    val_mean,val_std=[],[]
    for k in metrics_dict['val']:
        val_mean.append(np.mean(metrics_dict['val'][k]))
        val_std.append(np.std(metrics_dict['val'][k]))
    print(f"|AUROC| SNS: {val_mean[0]:.4f}+-{val_std[0]:.4f} | MNS: {val_mean[2]:.4f}+-{val_std[2]:.4f} | CNS: {val_mean[4]:.4f}+-{val_std[4]:.4f} | MIX: {val_mean[6]:.4f}+-{val_std[6]:.4f} | AVG: {val_mean[8]:.4f}+-{val_std[8]:.4f}")
    print(f"|AP| SNS: {val_mean[1]:.4f}+-{val_std[1]:.4f} | MNS: {val_mean[3]:.4f}+-{val_std[3]:.4f} | CNS: {val_mean[5]:.4f}+-{val_std[5]:.4f} | MIX: {val_mean[7]:.4f}+-{val_std[7]:.4f} | AVG: {val_mean[9]:.4f}+-{val_std[9]:.4f}")

    print('---------Test set-----------')
    test_mean,test_std=[],[]
    for k in metrics_dict['test']:
        test_mean.append(np.mean(metrics_dict['test'][k]))
        test_std.append(np.std(metrics_dict['test'][k]))
    print(f"|AUROC| SNS: {test_mean[0]:.4f}+-{test_std[0]:.4f} | MNS: {test_mean[2]:.4f}+-{test_std[2]:.4f} | CNS: {test_mean[4]:.4f}+-{test_std[4]:.4f} | MIX: {test_mean[6]:.4f}+-{test_std[6]:.4f} | AVG: {test_mean[8]:.4f}+-{test_std[8]:.4f}")
    print(f"|AP| SNS: {test_mean[1]:.4f}+-{test_std[1]:.4f} | MNS: {test_mean[3]:.4f}+-{test_std[3]:.4f} | CNS: {test_mean[5]:.4f}+-{test_std[5]:.4f} | MIX: {test_mean[7]:.4f}+-{test_std[7]:.4f} | AVG: {test_mean[9]:.4f}+-{test_std[9]:.4f}")

def evaluate_edge_loader(model, data, dataloader):
    model.eval()
    test_preds, test_labels = [], []

    with torch.no_grad():
        
        while True:
            # 1.message passing
            n,_ = model.encoding(data)

            # 2. candidate scoring for hyperedge in validation/test datasets
            hedges, labels, is_last = dataloader.next() 
            test_preds += model.aggregate(n, hedges, mode='Eval') 
            test_labels.append(labels.detach())
            
            if is_last:
                break

        test_preds = torch.sigmoid(torch.stack(test_preds).view(-1))
        test_labels = torch.cat(test_labels, dim=0)

    return test_preds.tolist(), test_labels.tolist()

def eval_edge_train(model,data,batch_loaders):
    
    train_pred_pos, train_label_pos = evaluate_edge_loader(model, data, batch_loaders['train_pos'])
    train_pred_neg, train_label_neg = evaluate_edge_loader(model, data, batch_loaders['train_neg'])
    
    roc_train = roc_auc_score(np.array(train_label_pos+train_label_neg), np.array(train_pred_pos+train_pred_neg))
    ap_train = average_precision_score(np.array(train_label_pos+train_label_neg), np.array(train_pred_pos+train_pred_neg))
    
    metrics_dict={
        'roc_train':roc_train,
        'ap_train':ap_train
    }
    
    return metrics_dict

def eval_edge_val_test(model,data,batch_loaders,mode='val'):
    
    val_pred_pos, val_label_pos = evaluate_edge_loader(model, data, batch_loaders[f'{mode}_pos'])
    val_pred_sns, val_label_sns = evaluate_edge_loader(model, data, batch_loaders[f'{mode}_neg_sns'])
    val_pred_mns, val_label_mns = evaluate_edge_loader(model, data, batch_loaders[f'{mode}_neg_mns'])
    val_pred_cns, val_label_cns = evaluate_edge_loader(model, data, batch_loaders[f'{mode}_neg_cns'])
    
    # SNS set
    roc_sns = roc_auc_score(np.array(val_label_pos+val_label_sns), np.array(val_pred_pos+val_pred_sns))
    ap_sns = average_precision_score(np.array(val_label_pos+val_label_sns),np.array(val_pred_pos+val_pred_sns))

    # MNS set
    roc_mns = roc_auc_score(np.array(val_label_pos+val_label_mns), np.array(val_pred_pos+val_pred_mns))
    ap_mns = average_precision_score(np.array(val_label_pos+val_label_mns),np.array(val_pred_pos+val_pred_mns))

    # CNS set
    roc_cns = roc_auc_score(np.array(val_label_pos+val_label_cns), np.array(val_pred_pos+val_pred_cns))
    ap_cns = average_precision_score(np.array(val_label_pos+val_label_cns),np.array(val_pred_pos+val_pred_cns))

    # Mixed set
    d = len(val_pred_pos) // 3
    val_label_mixed = val_label_pos + val_label_sns[0:d]+val_label_mns[0:d]+val_label_cns[0:d]
    val_pred_mixed = val_pred_pos + val_pred_sns[0:d]+val_pred_mns[0:d]+val_pred_cns[0:d]
    roc_mixed = roc_auc_score(np.array(val_label_mixed), np.array(val_pred_mixed))
    ap_mixed = average_precision_score(np.array(val_label_mixed),np.array(val_pred_mixed))

    roc_average = (roc_sns+roc_mns+roc_cns+roc_mixed)/4
    ap_average = (ap_sns+ap_mns+ap_cns+ap_mixed)/4

    metrics_dict={
        'roc_sns':roc_sns,
        'ap_sns':ap_sns,
        'roc_mns':roc_mns,
        'ap_mns':ap_mns,
        'roc_cns':roc_cns,
        'ap_cns':ap_cns,
        'roc_mixed':roc_mixed,
        'ap_mixed': ap_mixed,
        'roc_average':roc_average,
        'ap_average':ap_average
    }
    
    return metrics_dict

def fair_metric(pred, labels, sens):

    idx_s0 = sens == 0
    idx_s1 = sens == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)

    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) -
                 sum(pred[idx_s1]) / sum(idx_s1))

    equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) -
                   sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
    return parity.item()*100, equality.item()*100

def masked_accuracy(logits: Tensor, labels: Tensor):
    if len(logits) == 0:
        return 0
    pred = torch.argmax(logits, dim=1)
    acc = pred.eq(labels).sum() / len(logits) * 100
    return acc.item()

def accuracy(logits: Tensor, labels: Tensor, masks: dict[Tensor]):
    accs = []
    for mask in masks.values():
        acc = masked_accuracy(logits[mask], labels[mask])
        accs.append(acc)
    return accs

def masked_f1_score(logits: Tensor, labels: Tensor):
    if len(logits) == 0:
        return 0
    pred = torch.argmax(logits, dim=1)
    f1=f1_score(labels.cpu().numpy(), pred.cpu().numpy()) 
    return f1

def f1_scores(logits: Tensor, labels: Tensor, masks: dict[Tensor]):
    f1s = []
    for mask in masks.values():
        f1 = masked_f1_score(logits[mask], labels[mask])
        f1s.append(f1)
    return f1s

def masked_auc_roc(auc_score: Tensor, labels: Tensor):
    if len(auc_score) == 0:
        return 0
    auc_roc=roc_auc_score(labels.cpu().numpy(), auc_score.cpu().numpy())
    return auc_roc

def auc_rocs(logits: Tensor, labels: Tensor, masks: dict[Tensor]):

    probs = F.softmax(logits, dim=1)
    auc_score = probs[:, 1].detach()  
    aucs = []
    for mask in masks.values():
        auc = masked_auc_roc(auc_score[mask], labels[mask])
        aucs.append(auc)
    return aucs

def masked_fairness(logits: Tensor, labels: Tensor, sens):

    if len(logits) == 0:
        return 0
    pred = torch.argmax(logits, dim=1)
    dp,eo = fair_metric(pred.cpu().numpy(), labels.cpu().numpy(), sens.cpu().numpy())
    return dp,eo

def fairness(logits: Tensor, labels: Tensor, masks: dict[Tensor], sens):
    dps,eos = [],[]
    for mask in masks.values():
        dp,eo = masked_fairness(logits[mask], labels[mask], sens[mask])
        dps.append(dp)
        eos.append(eo)
    return dps,eos
