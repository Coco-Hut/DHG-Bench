import argparse
import os
import yaml
from lib_dataset import _single_datasets_,_multi_datasets_

def update_from_dict(obj, updates):
    for key, value in updates.items():
        # set higher priority from command line as we explore some factors
        if key in ['init'] and obj.init is not None:
            continue
        setattr(obj, key, value)

# recommend hyperparameters here
def method_config(args):

    if args.is_default:
        config_name = 'default'
    else:
        config_name = args.dname
    try:
        # conf_dt = json.load(open(f"{os.path.join('./', 'lib_configs', args.method.lower(), config_name)}.json")) 
        task_prefix=args.task_type.split('_')[0]+'_yamls'
        conf_dt = yaml.safe_load(open(f"{os.path.join('./', 'lib_yamls', task_prefix,'config_'+args.method.lower())}.yaml"))[config_name] 
        update_from_dict(args, conf_dt)
    except:
        print('No config file found or error in json format, please use method_config(args)')

    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def validate_observed_edge_prediction_args(args):
    is_perturbed = getattr(args, 'is_perturbed', False)
    if isinstance(is_perturbed, str):
        is_perturbed = str2bool(is_perturbed)
        args.is_perturbed = is_perturbed

    if (
        args.task_type == 'edge_pred'
        and getattr(args, 'edge_pred_protocol', 'legacy') == 'observed'
        and is_perturbed
    ):
        raise ValueError(
            'Perturbations are not supported with '
            '--edge_pred_protocol=observed; disable --is_perturbed or use '
            '--edge_pred_protocol=legacy'
        )

def set_task_args(args):
    validate_observed_edge_prediction_args(args)
    
    if args.task_type == 'node_cls':
        if args.dname not in _single_datasets_:
            raise ValueError('The dataset is not suitable for node classification')
        args.add_self_loop=True 
        if args.use_bench_prop:
            args.train_prop,args.valid_prop = 0.5,0.25
        args.early_stop = False
    elif args.task_type == 'hg_cls':
        if args.dname not in _multi_datasets_:
            raise ValueError('The datasets is not suitable for hypergraph classification')
        args.add_self_loop=False
        if args.use_bench_prop:
            args.train_prop,args.valid_prop = 0.8,0.1
        args.early_stop = True
        if args.method in ['EHNN','TMPHN']:
            raise ValueError(f'{args.method} is not supoorted for hypergraph classification task') 
    else:
        if args.dname not in _single_datasets_:
            raise ValueError('The dataset is not suitable for edge prediction')

        if args.method in ['HyperND']:
            args.add_self_loop=True 
        elif args.method in ['DPHGNN','LEGCN','PhenomNN','HJRL','TFHNN','HNHN','AllSetformer'] and args.dname in ['pokec']:
            args.add_self_loop=True
        elif args.method in ['TMPHN']:
            if args.dname in ['pokec']:
                args.add_self_loop=True
            else:
                args.add_self_loop=False
            args.device='cpu' 
        else:
            args.add_self_loop=False
        if args.use_bench_prop:
            args.train_prop,args.valid_prop = 0.6,0.2
        args.early_stop = True
    
    return args

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyper-parameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser()

    ######################### general parameters ################################
    '''
    Semi-supervised setting: Train/Valid/Test: 50/25/25
    
    '''
    parser.add_argument('--use_bench_prop', default=True)
    parser.add_argument('--train_prop', type=float, default=0.6)
    parser.add_argument('--valid_prop', type=float, default=0.2)

    parser.add_argument('--dname', default='cora',choices=['cora','citeseer','pubmed',
                                                            'coauthor_cora','coauthor_dblp',
                                                            '20newsW100', 'ModelNet40', 'zoo','NTU2012', 'Mushroom',
                                                            'yelp','walmart-trips-100','house-committees-100',
                                                            'actor','amazon','pokec','twitch',
                                                            'german','bail','credit',
                                                            'amazon_review','magpm','trivago','ogbn_mag',
                                                            "RHG_3", "RHG_10", "RHG_table", "RHG_pyramid",
                                                            "IMDB_dir_form", "IMDB_dir_genre",
                                                            "IMDB_wri_form", "IMDB_wri_genre",
                                                            "stream_player","twitter_friend"])
    
    parser.add_argument('--task_type',default='edge_pred',choices=['node_cls','edge_pred','hg_cls'])
    parser.add_argument('--is_default',default=False)
    parser.add_argument('--use_processed', default=True)
    parser.add_argument('--method', default='HGNN') 
    
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_seeds', type=int, default=2)
    parser.add_argument('--epochs', default=5, type=int) 
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.001, type=float) # []
    parser.add_argument('--wd', default=0.0, type=float)
    parser.add_argument('--clip_grad',default=False,type=bool)
    parser.add_argument('--clip_thresh',default=5.0,type=float)
    parser.add_argument('--num_splits',type=int,default=10)
    parser.add_argument('--mem_verbose',default=True)
    parser.add_argument('--mem_display_step',default=100)
    parser.add_argument('--display_step', type=int, default=20)
    parser.add_argument('--eval_verbose',default=True)
    
    parser.add_argument('--embedding_mode', default=False, type=str2bool)
    parser.add_argument('--embedding_hidden',default=128,type=int) 
    
    parser.add_argument('--normtype', default='all_one') # ['all_one','deg_half_sym']
    parser.add_argument('--add_self_loop', action='store_false')
    parser.add_argument('--exclude_self', action='store_true')
    
    parser.add_argument('--edge_split_mode',default='ind',choices=['ind','trand'])
    parser.add_argument('--edge_pred_protocol',default='legacy',choices=['legacy','observed'])
    parser.add_argument('--edge_save_dir', action='store_true',default=f'./lib_edge_splits/') 
    parser.add_argument('--edge_batch_size', action='store_true',default=512) 
    parser.add_argument('--e_embed_hidden',default=64) 
    parser.add_argument('--e_embed_layer',default=2)
    parser.add_argument('--e_embed_dropout',default=0.2) 
    parser.add_argument('--e_embed_norm',default='ln') 
    parser.add_argument('--aggr_mode',default='max',choices=['max','mean','maxmin'])
    parser.add_argument('--ns_method',default='mixed',choices=['mns','sns','cns','mixed']) 
    parser.add_argument('--edge_aggr',default='group',choices=['group','single'])
    
    parser.add_argument('--hg_batch_size',default=256) # batch_size
    parser.add_argument('--pooling',default='mean')
    parser.add_argument('--g_embed_hidden',default=128) 
    parser.add_argument('--g_embed_layer',default=2) 
    parser.add_argument('--g_embed_dropout',default=0.2) 
    parser.add_argument('--g_embed_norm',default='ln') 
    parser.add_argument('--use_weighted_loss',default=False)
    parser.add_argument('--early_stop',default=True) 

    parser.add_argument('--is_perturbed',default=False) 
    parser.add_argument('--is_poison',default=True) 
    parser.add_argument('--pert_mode',default='spar_label',choices=['spar_feat','noise_feat',
                                                                    'drop_incidence','add_incidence',
                                                                    'spar_label','flip_label'])
    parser.add_argument('--pert_p',default=0.0) 

    # Choose std for synthetic feature noise
    parser.add_argument('--feature_noise', default='0.6', type=str)
    
    parser.set_defaults(add_self_loop=False)
    parser.set_defaults(exclude_self=False)

    args = parser.parse_args()

    return args
