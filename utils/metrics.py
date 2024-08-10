import os
import torch
import torch.nn as nn
import numpy as np
import math
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import concurrent

cpu_cores = os.cpu_count() // 2

class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count

class Metric(object):
    def __init__(
            self,
            max_seq_len = 12,
    ):
        self.max_seq_len = max_seq_len
        self.etr_mae = AverageMeter()
        self.same_rate = AverageMeter()
        self.online_etr_mae = AverageMeter()
        self.online_same_rate = AverageMeter()
        self.cnt_ratio = AverageMeter()
        self.rank_same_rate = AverageMeter()
        self.online_rank_same_rate = AverageMeter()
        self.same_200 = AverageMeter()
        self.same_1 = AverageMeter()
        self.same_500 = AverageMeter()
        self.online_same_200 = AverageMeter()
        self.online_same_1 = AverageMeter()
        self.online_same_500 = AverageMeter()

        self.hr = [AverageMeter() for _ in range(self.max_seq_len)]
        self.lsd = AverageMeter()
        self.krc = AverageMeter()
        self.lmd = AverageMeter()
        self.ed = AverageMeter() #edit distance
        self.acc = [AverageMeter() for _ in range(self.max_seq_len)]

        self.online_hr = [AverageMeter() for _ in range(self.max_seq_len)]
        self.online_lsd = AverageMeter()
        self.online_krc = AverageMeter()
        self.online_lmd = AverageMeter()
        self.online_ed = AverageMeter()
        self.online_acc = [AverageMeter() for _ in range(self.max_seq_len)]


    def eval_metrics(self, labels, train_results, flags,geohash_dist_mat, is_test):

        self.is_test = is_test
        # krc =  np.array([kendall_rank_correlation(pre, lab, lab_len) for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
        # self.krc.update(krc, batch_size)

        batch_size, graph_size = train_results["selections"].shape
        labels = labels.view(batch_size, graph_size, 5)
        (target_etr_predictions, online_target_etr_predictions, target_etr_labels, 
         same_rate, rank_same_rate, online_same_rate, 
         online_rank_same_rate, predict_mask_int,sr_200, sr_1, sr_500, online_sr_200, online_sr_1, online_sr_500) = self.eval_stat(labels, train_results, flags,geohash_dist_mat)

        # Calculate Mean Absolute Error for ETR
        # etr_mae = torch.abs(target_etr_labels / 60.0 - target_etr_predictions / 60.0).mean()
        # online_etr_mae = torch.abs(target_etr_labels / 60.0 - online_target_etr_predictions / 60.0).mean()
        # same_rate = same_rate.mean()
        # online_same_rate = online_same_rate.mean()
        # cnt_ratio = predict_mask_int.mean()
        self.etr_mae.update(torch.abs(target_etr_labels / 60.0 - target_etr_predictions / 60.0).mean(), batch_size)
        self.online_etr_mae.update(torch.abs(target_etr_labels / 60.0 - online_target_etr_predictions / 60.0).mean(), batch_size)
        self.same_rate.update(same_rate, batch_size)
        self.online_same_rate.update(online_same_rate, batch_size)
        self.cnt_ratio.update(predict_mask_int.mean(),predict_mask_int.numel())
        self.rank_same_rate.update(rank_same_rate, batch_size)
        self.online_rank_same_rate.update(online_rank_same_rate, batch_size)

        self.same_200.update(sr_200, batch_size)
        self.same_1.update(sr_1, batch_size)
        self.same_500.update(sr_500, batch_size)
        self.online_same_200.update(online_sr_200, batch_size)
        self.online_same_1.update(online_sr_1, batch_size)
        self.online_same_500.update(online_sr_500, batch_size)

    def eval_sr(self, pred, label, point_mat, threshold):

        # 计算有效的标签数量（不等于 -1 的元素数量）
        valid_labels = (label != -1)
        label_v = valid_labels.sum(dim=1)

        # 计算所有有效标签的总数
        count_all = label_v.sum()

        # 检查预测是否正确或点矩阵的值是否小于阈值
        correct_or_below_threshold = (pred == label) | (point_mat[torch.arange(pred.shape[0]).unsqueeze(1), pred+2, label+2] < threshold)

        # 只考虑有效的标签位置
        correct_or_below_threshold = correct_or_below_threshold & valid_labels

        # 创建一个掩码，一旦条件不满足就停止考虑该行的后续元素
        cumulative_mask = torch.cumprod(correct_or_below_threshold, dim=1)

        # 计算所有行的正确预测总数
        count = cumulative_mask.sum()
        # 计算结果
        result = count / count_all if count_all != 0 else 0

        return result
    
    def eval_lsd(self, pred, label, label_len, mode='square'):

        def _sigmoid(x):
            s = 1 / (1 + np.exp(-x))
            return s

        def idx_weight(i, mode='linear'):
            if mode == 'linear': return 1 / (i + 1)
            if mode == 'exp': return math.exp(-i)
            if mode == 'sigmoid': return _sigmoid(5 - i)  # 5 means we focuse on the top 5
            if mode == 'no_weight': return 1
            if mode == 'log': return 1 / math.log(2 + i)  # i is start from 0
        """
        calculate LSD / LMD
        mode:
        'square', The Location Square Deviation (LSD)
            else:    The Location Mean Deviation (LMD)
        """
        label = label[:label_len]

        n = len(label)
        # get the location in list 1
        idx_1 = [idx for idx, x in enumerate(label)]
        # get the location in list 2
        for i in range(len(label)):
            if label[i] not in pred:
                # return 2
                print(pred)
                print(label)
        idx_2 = [pred.index(x) for x in label]

        # caculate the distance
        idx_diff = [math.fabs(i - j) for i, j in zip(idx_1, idx_2)]
        weights = [idx_weight(idx, 'no_weight') for idx in idx_1]

        result = list(map(lambda x: x ** 2, idx_diff)) if mode == 'square' else idx_diff
        return sum([diff * w for diff, w in zip(result, weights)]) / n
    


    def edit_distance(self, pred, label, label_len):

        """
        calculate edit distance (ED)
        """
        label = label[:label_len]
        import edit_distance
        assert set(label).issubset(set(pred)), "error in prediction"
        # Focus on the items in the label
        if not isinstance(pred, list): pred = pred.tolist()
        if not isinstance(label, list): label = label.tolist()
        try:
            pred = [x for x in pred if x in label]
            ed = edit_distance.SequenceMatcher(pred, label).distance()
        except:
            print('pred in function:', pred, f'type of pred: {type(pred)}')
            print('label in function:', label, f'type label:{type(label)}')
        return ed


    def eval_krc(self, pred, label, label_len):

        """
        caculate  kendall rank correlation (KRC), note that label set is a subset of pred set
        """
        def is_concordant(i, j):
            return 1 if (label_order[i] < label_order[j] and pred_order[i] < pred_order[j]) or (
                    label_order[i] > label_order[j] and pred_order[i] > pred_order[j]) else 0

        if label_len == 1: return 1

        label = label[:label_len]
        not_in_label = set(pred) - set(label)# 0
        # get order dict
        pred_order = {d: idx for idx, d in enumerate(pred)}
        label_order = {d: idx for idx, d in enumerate(label)}
        for o in not_in_label:
            label_order[o] = len(label)

        n = len(label)
        # compare list 1: compare items between labels
        lst1 = [(label[i], label[j]) for i in range(n) for j in range(i + 1, n)]
        # compare list 2: compare items between label and pred
        lst2 = [(i, j) for i in label for j in not_in_label]

        try:
            hit_lst = [is_concordant(i, j) for i, j in (lst1 + lst2)]
        except:
            print('[warning]: wrong in calculate KRC')
            return float(1)
            # return float(0)

        hit = sum(hit_lst)
        not_hit = len(hit_lst) - hit
        result = (hit - not_hit) / (len(lst1) + len(lst2))
        return result
    
    def eval_hr(self, pred, label, lab_len, top_n=3):

        """
        calculate Hit-Rate@k (HR@k)
        """
        eval_num = min(top_n, lab_len)
        hit_num = len(set(pred[:eval_num]) & set(label[:eval_num]))
        hit_rate = hit_num / eval_num
        return hit_rate

    def eval_acc(self, pred, label, label_len, top_n):
        
        """
        calculate ACC@k
        """
        assert set(label[:label_len]).issubset(set(pred)), f"error in prediction:{pred}, label:{label}"
        eval_num = min(top_n, label_len)
        pred = pred[:eval_num]
        label = label[:eval_num]
        if not isinstance(pred, list): pred = pred.tolist()
        if not isinstance(label, list): label = label.tolist()
        for i in range(eval_num):# which means the sub route should be totally correct.
            if not pred[i] == label[i]: return 0
        return 1

    def eval_stat(self, labels, train_results, flags, point_mat):
        
        batch_size, graph_size, _ = labels.shape
        online_etr_array = labels[:, :, 3].float()
        online_rp_array = labels[:, :, 2]
        predict_mask = train_results["label_selections"] >= 0
        predict_mask_float = predict_mask.float()
        label_selections = train_results["label_selections"].reshape(-1)
        etr_selections = torch.where(label_selections >= 0)[0].view(-1) 
        etr_labels = train_results["label_etr"].reshape(-1).float()
        target_etr_labels = etr_labels[etr_selections]
        # 预估ETR
        etr_predictions = train_results["etr"].view(batch_size, graph_size, 9)
        etr_predictions = etr_predictions[:, :, 4].reshape(-1)
        target_etr_predictions = etr_predictions[etr_selections]

        # 预估RP
        predict_same = ((train_results["label_selections"] == train_results["selections"]) & predict_mask).long()
        cum_predict_same = predict_same.cumprod(dim=1).float()
        # cum_predict_same = predict_same
        same_rate = cum_predict_same.sum() / (predict_mask_float.sum() + 1e-6)
        rank_same_rate = cum_predict_same.sum(dim=0) / (predict_mask_float.sum(dim=0) + 1e-6)
        # 线上ETR&RP
        online_selections = []
        online_etr = []
        for i in range(graph_size):
            label_idx = (online_rp_array - 2 == i).long()
            max_idx = label_idx.argmax(dim=1)
            max_idx = torch.where(torch.all(label_idx <= 0, dim=1), torch.ones_like(max_idx) * -1, max_idx)
            online_selections.append(max_idx)
            current_selection = max_idx.view(batch_size, -1)
            current_selection = torch.where(current_selection < 0,
                                            torch.ones_like(current_selection) * (graph_size - 1), current_selection)
            selection_indices = torch.cat(
                [torch.arange(batch_size, dtype=torch.long).view(batch_size, -1).to(online_etr_array.device), current_selection], dim=-1)
            online_etr.append(online_etr_array[selection_indices[:, 0], selection_indices[:, 1]])
        online_selections = torch.stack(online_selections, dim=1)
        online_etr = torch.stack(online_etr, dim=1)
        # 线上ETR
        online_etr_predictions = online_etr.view(-1)
        online_target_etr_predictions = online_etr_predictions[etr_selections]
        # 线上RP
        online_predict_same = ((train_results["label_selections"] == online_selections) & predict_mask).long()
        online_cum_predict_same = online_predict_same.cumprod(dim=1).float()
        # online_cum_predict_same = online_predict_same.float()
        online_same_rate = online_cum_predict_same.sum() / (predict_mask_float.sum() + 1e-6)
        online_rank_same_rate = online_cum_predict_same.sum(dim=0) / (predict_mask_float.sum(dim=0) + 1e-6)

        pred = train_results["selections"]
        label = train_results["label_selections"]

        sr_200 = self.eval_sr(pred, label, point_mat, 200)
        sr_1 = self.eval_sr(pred, label, point_mat, 1)
        sr_500 = self.eval_sr(pred, label, point_mat, 500)
        online_sr_200 = self.eval_sr(online_selections, label, point_mat, 200)
        online_sr_1 = self.eval_sr(online_selections, label, point_mat, 1)
        online_sr_500 = self.eval_sr(online_selections, label, point_mat, 500)


        if self.is_test:

            pred = pred.cpu().numpy().tolist()
            label = label.cpu().numpy().tolist()
            online_selections = online_selections.cpu().numpy().tolist()


            label_lens = (np.array(label) >= 0).sum(axis=1).tolist()

            for n in range(3):
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
                    hr_n = executor.map(self.eval_hr, pred, label, label_lens, [n + 1] * batch_size)

                hr_n = np.array(list(hr_n)).mean()
                self.hr[n].update(hr_n, batch_size)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
                krc = executor.map(self.eval_krc, pred, label, label_lens)
            krc = np.array(list(krc)).mean()
            self.krc.update(krc, batch_size)

            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
                lsd = executor.map(self.eval_lsd, pred, label, label_lens)
            lsd = np.array(list(lsd)).mean()
            self.lsd.update(lsd, batch_size)

            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
                lmd = executor.map(self.eval_lsd, pred, label, label_lens, ['lmd'] * batch_size)
            lmd = np.array(list(lmd)).mean()
            self.lmd.update(lmd, batch_size)

            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
                ed = executor.map(self.edit_distance, pred, label, label_lens)
            ed = np.array(list(ed)).mean()
            self.ed.update(ed, batch_size)

            for n in range(3):
                with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
                    acc_n = executor.map(self.eval_acc, pred, label, label_lens, [n + 1] * batch_size)
                acc_n = np.array(list(acc_n)).mean()
                self.acc[n].update(acc_n, batch_size)
            
            for n in range(3):

                with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
                    hr_n = executor.map(self.eval_hr, online_selections, label, label_lens, [n + 1] * batch_size)
                hr_n = np.array(list(hr_n)).mean()
                self.online_hr[n].update(hr_n, batch_size)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
                online_krc = executor.map(self.eval_krc, online_selections, label, label_lens)
            online_krc = np.array(list(online_krc)).mean()
            self.online_krc.update(online_krc, batch_size)

            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
                online_lsd = executor.map(self.eval_lsd, online_selections, label, label_lens)
            online_lsd = np.array(list(online_lsd)).mean()
            self.online_lsd.update(online_lsd, batch_size)

            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
                online_lmd = executor.map(self.eval_lsd, online_selections, label, label_lens, ['lmd'] * batch_size)
            online_lmd = np.array(list(online_lmd)).mean()
            self.online_lmd.update(online_lmd, batch_size)

            
            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
                online_ed = executor.map(self.edit_distance, online_selections, label, label_lens)
            online_ed = np.array(list(online_ed)).mean()
            self.online_ed.update(online_ed, batch_size)


            for n in range(3):
                with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
                    acc_n = executor.map(self.eval_acc, online_selections, label, label_lens, [n + 1] * batch_size)
                acc_n = np.array(list(acc_n)).mean()
                self.online_acc[n].update(acc_n, batch_size)
    
        return target_etr_predictions, online_target_etr_predictions, target_etr_labels, \
            same_rate, rank_same_rate, online_same_rate, online_rank_same_rate, predict_mask_float, \
            sr_200, sr_1, sr_500, online_sr_200, online_sr_1, online_sr_500


    def out(self):

        if self.is_test:
            metrics = {
                "etr_mae": self.etr_mae.avg.item(),  # Converting to Python scalar
                "online_etr_mae": self.online_etr_mae.avg.item(),
                "same_rate": self.same_rate.avg.item(),  # or same_rate_mean if it's a tensor
                "online_same_rate": self.online_same_rate.avg.item(),  # or online_same_rate_mean
                "cnt_ratio": self.cnt_ratio.avg.item(),
                "rank_sr_1": self.rank_same_rate.avg[0].item(),
                "online_sr_1": self.online_rank_same_rate.avg[0].item(),
                "rank_sr_2": self.rank_same_rate.avg[1].item(),
                "online_sr_2": self.online_rank_same_rate.avg[1].item(),           
                "rank_sr_3": self.rank_same_rate.avg[2].item(),
                "online_sr_3": self.online_rank_same_rate.avg[2].item(), 
                "same_sr200": self.same_200.avg.item(),
                "same_sr1": self.same_1.avg.item(),
                "same_sr500": self.same_500.avg.item(),
                "online_same_sr200": self.online_same_200.avg.item(),
                "online_same_sr1": self.online_same_1.avg.item(),
                "online_same_sr500": self.online_same_500.avg.item(),
                "krc": self.krc.avg.item(),
                "lmd": self.lmd.avg.item(),
                "lsd": self.lsd.avg.item(),
                "ed": self.ed.avg.item(),
                "hr@1": self.hr[0].avg.item(),
                "hr@2": self.hr[1].avg.item(),
                "hr@3": self.hr[2].avg.item(),
                "acc@1": self.acc[0].avg.item(),
                "acc@2": self.acc[1].avg.item(),
                "acc@3": self.acc[2].avg.item(),
                "online_hr@1": self.online_hr[0].avg.item(),
                "online_hr@2": self.online_hr[1].avg.item(),
                "online_hr@3": self.online_hr[2].avg.item(),
                "online_acc@1": self.online_acc[0].avg.item(),
                "online_acc@2": self.online_acc[1].avg.item(),
                "online_acc@3": self.online_acc[2].avg.item(),
                "online_krc": self.online_krc.avg.item(),
                "online_lmd": self.online_lmd.avg.item(),
                "online_lsd": self.online_lsd.avg.item(),
                "online_ed": self.online_ed.avg.item(),
            }
        else:
            metrics = {
                "etr_mae": self.etr_mae.avg.item(),  # Converting to Python scalar
                "online_etr_mae": self.online_etr_mae.avg.item(),
                "same_rate": self.same_rate.avg.item(),  # or same_rate_mean if it's a tensor
                "online_same_rate": self.online_same_rate.avg.item(),  # or online_same_rate_mean
                "cnt_ratio": self.cnt_ratio.avg.item(),
                "rank_sr_1": self.rank_same_rate.avg[0].item(),
                "online_sr_1": self.online_rank_same_rate.avg[0].item(),
                "rank_sr_2": self.rank_same_rate.avg[1].item(),
                "online_sr_2": self.online_rank_same_rate.avg[1].item(),           
                "rank_sr_3": self.rank_same_rate.avg[2].item(),
                "online_sr_3": self.online_rank_same_rate.avg[2].item(), 
                "same_sr200": self.same_200.avg.item(),
                "same_sr1": self.same_1.avg.item(),
                "same_sr500": self.same_500.avg.item(),
                "online_same_sr200": self.online_same_200.avg.item(),
                "online_same_sr1": self.online_same_1.avg.item(),
                "online_same_sr500": self.online_same_500.avg.item(),
            }


        metrics = {k: round(v, 4) for k, v in metrics.items()}
        return metrics