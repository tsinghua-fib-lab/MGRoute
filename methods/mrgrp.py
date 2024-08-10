# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
import fcntl
import models
from utils.functions import EarlyStopping, define_loss
from utils.metrics import Metric
from dataset.RPdataset import RPDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

def print_m(*args, **kwargs):
    # 设置flush为True，除非已经明确提供了其他值
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)

def mrgrp_train(args, flags, is_print=False, random_seed=729):
    
    is_test = False
    log_dir = os.path.join(args.log_dir, f'seed{random_seed}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    context_c_sizes = [flags.dict_length_day_of_week_c + 1, flags.dict_length_quarter_of_day_c,
                        flags.dict_length_tc_manage_type_c + 1]
    wb_c_sizes = [flags.dict_length_is_new_wb_c, flags.dict_length_is_prebook_c,
                flags.dict_length_delivery_service_c + 1, flags.dict_length_busi_source_c,
                flags.dict_length_time_type_c]
    pickup_c_sizes = [flags.dict_length_poi_familiarity_c + 1, flags.dict_length_poi_type_c,
                    flags.dict_length_pushmeal_before_dispatch_c, flags.dict_length_wifi_arrive_shop_c,
                    flags.dict_length_late_meal_report_c]

    train_dataset = RPDataset(args, flags, mode='train')
    # batch size 1024
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    val_dataset = RPDataset(args, flags, mode='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = models.MRGRP(context_c_sizes, wb_c_sizes, pickup_c_sizes,flags, args)
    model = model.to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=flags.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    graph_size = int(flags.route_seq_len)
    # 绘制loss曲线  
    train_loss_list = []
    val_loss_list = []
    # 绘制acc曲线
    val_acc_list = []
    
    os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
    model_save_path = os.path.join(log_dir, 'checkpoints')

    es = EarlyStopping(patience=5, verbose=True, path=os.path.join(model_save_path, 'ckpt.pt'), is_print=is_print)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        val_loss = 0

        if is_print:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        # for i, features_all in enumerate(train_loader):
            # features_all = features["features_all"].to(device)
        for i, features_all in enumerate(train_loader):
            # if i<=510:
            #     continue
            features_all = features_all.to(args.device)
            batch_size = features_all.shape[0]
            labels = features_all[:,624:684].view(batch_size, graph_size, 5).long()
            # print_m("features_all.shape:",features_all.shape)
            result_final = model(features_all)
            results = {"logits": result_final[:,:144].view(batch_size, graph_size,graph_size),
                "selections": result_final[:,144:156].long(),
                "label_selections": result_final[:,156:168].long(),
                "etr": result_final[:,168:276],
                "label_etr": result_final[:,276:288]}
            # , context_c_sizes, wb_c_sizes, pickup_c_sizes
            
            loss = define_loss(labels, results, flags, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if is_print:
                train_loader.set_description(f'Epoch[{epoch}/{args.epochs}] | train-mse={loss:.5f}')
            # if i%10 == 0:
            #     if is_print:
            #         print_m("Epoch: {}, Iter: {}, Loss: {}".format(epoch, i, loss.item()))
            
            train_loss += loss.item()
        train_loss_list.append(train_loss/len(train_loader))

        scheduler.step()
        model.eval()

        with torch.no_grad():
            graph_size = int(flags.route_seq_len)
            # 特征相关
            eval_metrics = Metric(graph_size)
            # for i, features_all in enumerate(val_loader):
                # features_all = features["features_all"].to(device)
            if is_print:
                val_loader = tqdm(val_loader, desc=f"Validation Epoch {epoch}/{args.epochs}")
            
            for i, (features_all) in enumerate(val_loader):
                features_all = features_all.to(args.device)
                batch_size = features_all.shape[0]
                labels = features_all[:,624:684].view(batch_size, graph_size, 5).long()
                geohash_dist_mat = features_all[:,883:1079].view(batch_size, graph_size + 2, graph_size + 2)
                # result_final = model(features_all)
                # result_final = model(batch_size, graph_size, context_n_features, context_c_features, wb_c_features, wb_time_features, pickup_n_features, pickup_c_features, deliver_n_features, da_n_features, geohash_dist_mat, line_dist_mat, angle_mat, wb_ids, point_types, route_labels, etr_labels)
                result_final = model(features_all)
                results = {"logits": result_final[:,:144].view(batch_size, graph_size,graph_size),
                    "selections": result_final[:,144:156].long(),
                    "label_selections": result_final[:,156:168].long(),
                    "etr": result_final[:,168:276],
                    "label_etr": result_final[:,276:288]}
                # Other Stat
                # Adapt the `eval_stat` method to PyTorch if necessary
                # metrics = self.define_eval_metrics(labels, results,flags)
                loss = define_loss(labels, results,flags, args)
                eval_metrics.eval_metrics(labels, results,flags,geohash_dist_mat, is_test)
                val_loss += loss.item()

            metrics = eval_metrics.out()
            if is_print:
                print_m("Epoch: {}, Iter_num: {}, Metrics: {}".format(epoch, i, metrics))
            val_loss_list.append(val_loss/len(val_loader))
            val_acc_list.append(metrics["same_sr200"])

            # early stopping
            es(metrics["same_sr200"], model)
            if es.early_stop:
                if is_print:
                    print_m("Early stopping")
                break
            
            # 绘制loss曲线
            plt.figure()
            plt.plot(train_loss_list, label='train_loss')
            plt.plot(val_loss_list, label='val_loss')
            plt.legend()
            plt.show()
            # plt.savefig("./saved/{}_".format(file_name) + str(exp_id) +"/loss.jpg")
            if not os.path.exists(os.path.join(log_dir, 'figs')):
                os.makedirs(os.path.join(log_dir, 'figs'), exist_ok=True)
            plt.savefig(os.path.join(log_dir, 'figs', 'loss.jpg'))
            # 绘制acc曲线
            plt.figure()
            plt.plot(val_acc_list, label='val_acc')
            plt.legend()
            plt.show()
            plt.savefig(os.path.join(log_dir, 'figs', 'same_sr200.jpg'))

def mrgrp_test(args, flags, is_print=False, random_seed=729):

    is_test = True

    log_dir = os.path.join(args.log_dir, f'seed{random_seed}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    model_save_path = os.path.join(log_dir, 'checkpoints')

    context_c_sizes = [flags.dict_length_day_of_week_c + 1, flags.dict_length_quarter_of_day_c,
                        flags.dict_length_tc_manage_type_c + 1]
    wb_c_sizes = [flags.dict_length_is_new_wb_c, flags.dict_length_is_prebook_c,
                flags.dict_length_delivery_service_c + 1, flags.dict_length_busi_source_c,
                flags.dict_length_time_type_c]
    pickup_c_sizes = [flags.dict_length_poi_familiarity_c + 1, flags.dict_length_poi_type_c,
                    flags.dict_length_pushmeal_before_dispatch_c, flags.dict_length_wifi_arrive_shop_c,
                    flags.dict_length_late_meal_report_c]
    
    model = models.MRGRP(context_c_sizes, wb_c_sizes, pickup_c_sizes,flags, args)
    model = model.to(args.device)
    try:
        model.load_state_dict(torch.load(os.path.join(model_save_path, 'ckpt.pt')))
    except:
        if is_print:
            print_m("Warning: No model checkpoint found, use initialized models......")

    test_dataset = RPDataset(args, flags, mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model.eval()
    test_loss = 0

    if is_print:
        test_loader = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        if is_print:
            print_m("Testing...")
        graph_size = int(flags.route_seq_len)
        eval_metrics = Metric(graph_size)
        for i, features_all in enumerate(test_loader):
            features_all = features_all.to(args.device)
            batch_size = features_all.shape[0]
            labels = features_all[:,624:684].view(batch_size, graph_size, 5).long()
            geohash_dist_mat = features_all[:,883:1079].view(batch_size, graph_size + 2, graph_size + 2)
            # result_final = model(features_all)
            # result_final = model(batch_size, graph_size, context_n_features, context_c_features, wb_c_features, wb_time_features, pickup_n_features, pickup_c_features, deliver_n_features, da_n_features, geohash_dist_mat, line_dist_mat, angle_mat, wb_ids, point_types, route_labels, etr_labels)
            result_final = model(features_all)
            results = {"logits": result_final[:,:144].view(batch_size, graph_size,graph_size),
                "selections": result_final[:,144:156].long(),
                "label_selections": result_final[:,156:168].long(),
                "etr": result_final[:,168:276],
                "label_etr": result_final[:,276:288]}
            # Other Stat
            # Adapt the `eval_stat` method to PyTorch if necessary
            # metrics = self.define_eval_metrics(labels, results,flags)
            loss = define_loss(labels, results,flags, args)
            test_loss += loss.item()
            eval_metrics.eval_metrics(labels, results,flags,geohash_dist_mat, is_test)

        test_loss = test_loss/len(test_loader)
        metrics = eval_metrics.out()
        if is_print:
            print_m("Test Loss: {}".format(test_loss))
            print_m("Test Metrics: {}".format(metrics))

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    
    with open(os.path.join(args.log_dir, 'test_metrics.txt'), 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        write_str = f'{random_seed}'
        for key, value in metrics.items():
            write_str += f',{value}'
        write_str += '\n'
        f.writelines(write_str)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
        
    

    





    


