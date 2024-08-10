# -*- coding: utf-8 -*-
import os
import time
import torch
import signal
import argparse
import pickle
from multiprocessing import Process
import multiprocessing
from utils.functions import seed_everything, set_cpu_num, AutoGPU, display_results
from dotmap import DotMap
import warnings

warnings.filterwarnings("ignore")

def subworker(args, flags, is_print=False, random_seed=729, mode='train', gpu_id=0):

    time.sleep(0.1)
    seed_everything(random_seed)
    set_cpu_num(1)

    if args.device == 'cuda':
        torch.cuda.set_device(gpu_id)

    if args.model == 'mrgrp':
        from methods import mrgrp_train as model_train
        from methods import mrgrp_test as model_test
        
    else:
        raise ValueError(f"Wrong model name {args.model}")
    
    if mode == 'train':
        model_train(args, flags, is_print, random_seed)
    elif mode == 'test':
        model_test(args, flags, is_print, random_seed)
    else:
        raise ValueError(f"Wrong mode {mode}")

def experiment(args, mode='train'):

    print(f'Start {mode}ing...')

    if os.path.exists(os.path.join(args.log_dir, 'test_metrics.txt')):
        os.remove(os.path.join(args.log_dir, 'test_metrics.txt'))
        
    flags_path = os.path.join(os.path.dirname(__file__), 'data', args.dataset_name, 'flags.pkl')

    flags = pickle.load(open(flags_path, 'rb'))
    flags = DotMap(flags)

    workers = []
    gpu_controller = AutoGPU(args.memory_size, args)

    for random_seed in range(1, args.seed_num+1):

        if args.parallel:
            is_print = True if (len(workers)==0 and mode=='train') else False
            gpu_id = gpu_controller.choice_gpu(args.force) if args.device == 'cuda' else 0 # select a gpu
            workers.append(Process(target=subworker, args=(args, flags, is_print, random_seed, mode, gpu_id), daemon=True))
            workers[-1].start()
            
        else:
            gpu_id = gpu_controller.choice_gpu(args.force) if args.device == 'cuda' else 0 # select a gpu
            subworker(args, flags, True, random_seed, mode, gpu_id)

    # block
    while args.parallel and any([sub.exitcode==None for sub in workers]):
        pass

    if mode == 'test':
        if os.path.exists(os.path.join(args.log_dir, 'results.txt')):
            os.remove(os.path.join(args.log_dir, 'results.txt'))
        display_results(args)

if __name__ == '__main__':

    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='beijing', help='dataset name')
    parser.add_argument('--model', type=str, default='mrgrp', help='model name')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--memory_size', type=int, default=9000, help='memory size')
    parser.add_argument('--seed_num', type=int, default=1, help='seed num')
    parser.add_argument('--log_dir', type=str, default='./logs', help='log dir')
    parser.add_argument('--parallel', action='store_true', help='parallel')
    parser.add_argument('--force', action='store_true', help='force')
    parser.add_argument('--cpu_num', type=int, default=1, help='cpu num')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--train_use_ratio', type=float, default=1.0, help='use train ratio')
    parser.add_argument('--val_use_ratio', type=float, default=1.0, help='use val ratio')
    parser.add_argument('--test_use_ratio', type=float, default=1.0, help='use test ratio')
    parser.add_argument('--dataset_type', type=str, default='')

    parser.add_argument('--test_only', action='store_true', help='test only')

    args = parser.parse_args()

    if args.dataset_type != '':
        args.dataset_name = f'{args.dataset_name}_{args.dataset_type}'
    args.log_dir = os.path.join(args.log_dir, args.model, args.dataset_name)

    if not args.parallel and args.cpu_num==1:
        print('[Warning] Not recommend to limit the cpu num when not parallel')
    
    if not args.test_only:
        experiment(args, 'train')
        
    experiment(args, 'test')



    






