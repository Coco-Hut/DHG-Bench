import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'dhgbench'))
from lib_utils.exp_agent import ExpAgent
from lib_models.HNN.preprocessing import algo_preprocessing
from lib_dataset.data_base import HyperDataset
from lib_dataset.preprocessing import data_processing
from parameter_parser import parameter_parser,method_config,set_task_args
from lib_dataset.data_perturbation import perturbation

if __name__ == '__main__':

    args = parameter_parser() 
    args = method_config(args) 
    args = set_task_args(args) 

    data=HyperDataset(args) 

    if args.task_type == 'hg_cls':
        data = data.multi_hypergraphs 
    else:
        data = data_processing(args,data)
        data._initialization_()

        if args.is_perturbed:
            if isinstance(args.pert_p,str):
                args.pert_p = eval(args.pert_p)
            if args.pert_mode not in ['spar_label','flip_label']:
                print('Robustness Perturbation for Structure and Feature')
                if args.is_poison:
                    print('Performing Poison Attack!')
                    data = perturbation(data,mode=args.pert_mode,p=args.pert_p,masks=None)
                else:
                    print('Performing Evasion Attack!')
                    evasion_data = perturbation(data,mode=args.pert_mode,p=args.pert_p,masks=None)
            else:
                print('Robustness Perturbation for Supervision Signal')
                if args.is_poison == False:
                    raise ValueError('Label attack is expected to be the poison attack!')

        data = algo_preprocessing(data,args)

        if args.is_perturbed and not args.is_poison:
            evasion_data = algo_preprocessing(evasion_data,args)
            data.evasion_data = evasion_data # store evasion data
    
    agent = ExpAgent(args)
    agent.running(args.task_type,data)