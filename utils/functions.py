import os
import torch
import random
import subprocess
import numpy as np
import fcntl
import time
import torch.nn as nn
import torch.nn.functional as F

def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    """
    Constructs token embedding matrix in PyTorch.
    vocab_size: scalar. V.
    num_units: embedding dimensionality. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero 如果 zero_pad 为真，则在嵌入矩阵的首行插入零填充。
    Returns
    weight variable: (V, E)
    """
    embeddings = nn.Embedding(vocab_size, num_units)
    nn.init.xavier_uniform_(embeddings.weight)

    if zero_pad:
        with torch.no_grad():
            embeddings.weight[0].fill_(0)
    
    return embeddings.weight
    # 特殊符号表示: 在自然语言处理（NLP）中，零填充通常用于表示特殊的符号，比如句子的开始或结束，或用于填充批处理中较短的句子。在嵌入矩阵中，索引0通常被保留用于这样的特殊符号。
    # 使用零填充（zero padding），即将嵌入矩阵的第一行（对应于特殊的填充词汇或符号）设置为零。
    # 如果不加with torch.no_grad()。 由于第一行被设为零，这个操作的梯度可能会尝试“修正”这个改变，导致在梯度更新时第一行的权重可能会被调整，从而不再保持为零。

def label_smoothing(inputs, epsilon=0.1):

    """
    这段代码定义了一个名为 label_smoothing 的函数，它实现了标签平滑（label smoothing）技术。
    标签平滑是一种在训练深度学习模型时常用的技术，用于增强模型的泛化能力，减少模型过于自信的预测（即模型对其预测的过度确定性）。
    Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    """
    v = inputs.size(-1)  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / v)


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    """
    Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    init_lr: 初始学习率，是一个标量值。
    global_step: 当前的训练步数，也是一个标量值。
    warmup_steps: 预热步数，是一个标量值。在这些预热步骤期间，学习率从一个较小的值增加到init_lr。
    步骤 调整global_step: step = tf.cast(global_step + 1, dtype=tf.float32) 将global_step转换为浮点数，并加1来避免在步数为0时除以0的情况。
    计算学习率:
    学习率是根据训练步数动态调整的。在预热阶段（即步数小于warmup_steps），学习率线性增加。达到warmup_steps后，学习率开始随着步数的平方根而减少。
    这通过以下公式实现：init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)。
    这个公式结合了一个线性增长项（step * warmup_steps ** -1.5）和一个衰减项（step ** -0.5），并取二者中的较小值。
    """
    step = global_step + 1
    return init_lr * warmup_steps ** 0.5 * min(step * warmup_steps ** -1.5, step ** -0.5)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, is_print=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.is_print = is_print
    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.is_print:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.is_print:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
    

def define_loss(labels, train_results, flags, args):
    batch_size, graph_size = train_results["selections"].shape
    # Rider's actual execution node
    label_selections = train_results["label_selections"].reshape(-1)
    point_selections = torch.where(label_selections >= 0)[0].view(-1)
    etr_labels = train_results["label_etr"].reshape(-1)
    target_etr_labels = etr_labels[point_selections]
    target_rp_labels = label_selections[point_selections]
    # ETR LOSS
    etr_predictions = train_results["etr"].reshape(batch_size*graph_size, 9) 
    target_etr_predictions = etr_predictions[point_selections]

    etr_loss = 0
    for i in range(9):
        quantile = (i + 1) * 0.1
        errors = target_etr_predictions[:, i] - target_etr_labels
        etr_loss += torch.mean(
            (1 - quantile) * torch.clamp(errors, min=0) + quantile * torch.clamp(-errors, min=0)
        )
    etr_loss = etr_loss / 60.0 / 9.0
    # loss = nn.L1Loss()

    # for i in range(9):
    #     tmp_quantile_mae = loss(target_etr_labels.float(), target_etr_predictions[:, i]) / 60.0
        # print_m("Quantile_{}_mae: {}".format(i, tmp_quantile_mae))
    # etr_loss = F.l1_loss(target_etr_labels, target_etr_predictions) / 60.0 / 5.0
    # Use tensorboard or other logging tools for logging
    # For example, using tensorboard: writer.add_scalar("etr_loss", etr_loss)
    # RP LOSS
    logit_predictions = train_results["logits"].reshape(-1, graph_size)
    target_logit_predictions = torch.index_select(logit_predictions, 0, point_selections)
    loss2 = nn.CrossEntropyLoss(size_average=True)
    # route_loss = F.cross_entropy(target_logit_predictions, target_rp_labels, reduction='mean')
    # softmax = nn.Softmax(dim=1)
    # nll = nn.NLLLoss()

    # print(target_logit_predictions)
    # print(target_rp_labels)

    # print(target_logit_predictions.shape)
    # print(target_rp_labels.shape)

    # print(target_logit_predictions[-1])
    # print(target_rp_labels[-1])
    # assert False
    
    route_loss = loss2(target_logit_predictions, target_rp_labels)

    # Use tensorboard or other logging tools for logging these metrics as well
    if args.ab_etr:
        return route_loss
    else:
        return route_loss + etr_loss * 0.1
    # return route_loss
    # sigma1 = F.softplus(log_sigma1)
    # sigma2 = F.softplus(log_sigma2)
    # weighted_loss = (1 / (2 * sigma1.pow(2)) * route_loss + 1 / (2 * sigma2.pow(2)) * etr_loss + torch.log(sigma1) + torch.log(sigma2))

    # return weighted_loss


def set_cpu_num(cpu_num):
    if cpu_num <= 0: return

    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AutoGPU():
    
    def __init__(self, memory_size, args):

        self.gpu_from_inside = args.gpu_from_inside
        
        self.memory_size = memory_size
        
        cmd = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode().strip().split("\n")

        self.free_memory = []
        for i, free_memory_str in enumerate(output):
            self.free_memory.append(int(free_memory_str))

    def update_free_memory(self):
        cmd = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode().strip().split("\n")
        self.free_memory = [int(free_memory_str) for free_memory_str in output]
    
    def get_required_memory(self, model_class, model_args, data):

        estimated_memory = 2000  # Start with an initial estimate (in MB)
        step = 500  # Increase step size (in MB)

        while True:
            try:
                # Try to load the model with the current estimated memory
                model = model_class(*model_args).cuda()
                dummy_input = torch.randn(*data).cuda()
                output = model(dummy_input)
                del model, dummy_input, output
                torch.cuda.empty_cache()
                break  # If successful, break the loop
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    estimated_memory += step
                else:
                    raise e  # Raise any other exceptions
        
        return estimated_memory
     
    def choice_gpu(self, force=False):


        available_gpu = []

        if not force:
            flag = False
            for i, free_memory in enumerate(self.free_memory):

                if free_memory >= self.memory_size:
                    
                    flag = True
                    available_gpu.append(i)
                    # self.free_memory[i] -= self.memory_size
                    # print(f"GPU-{i}: {free_memory}MB -> {self.free_memory[i]}MB")
                    
                    # return i
                
            if len(available_gpu) > 0:
                flag = True
                use_gpu = random.choice(available_gpu)
                self.free_memory[use_gpu] -= self.memory_size
                print(f"GPU-{use_gpu}: {self.free_memory[use_gpu]+self.memory_size}MB -> {self.free_memory[use_gpu]}MB")
                return use_gpu
                
            
            if not flag:
                print(f"SubProcess[{os.getpid()}]: No GPU can use, switch to CPU!")
                return -1
        else:
            while True:
                flag = False
                for i, free_memory in enumerate(self.free_memory):

                    if free_memory >= self.memory_size:
                        
                        flag = True
                        self.free_memory[i] -= self.memory_size
                        print(f"GPU-{i}: {free_memory}MB -> {self.free_memory[i]}MB")
                        
                        return i
                
                if not flag:
                    print(f"Waiting GPU......")
                    time.sleep(900)
                    self.update_free_memory()


def display_results(args):
    metrics_list = ['etr_mae', 'online_etr_mae', 'same_rate', 'online_same_rate', 'cnt_ratio', 'rank_sr_1', 'online_sr_1', 'rank_sr_2', 'online_sr_2', 'rank_sr_3', 'online_sr_3', 'same_sr200', 'same_sr1', 'same_sr500', 'online_same_sr200', 'online_same_sr1', 'online_same_sr500', 'krc', 'lmd', 'lsd', 'ed', 'hr@1', 'hr@2', 'hr@3', 'acc@1', 'acc@2', 'acc@3', 'online_hr@1', 'online_hr@2', 'online_hr@3', 'online_acc@1', 'online_acc@2', 'online_acc@3', 'online_krc', 'online_lmd', 'online_lsd', 'online_ed']
    metrics = np.zeros((args.seed_num, 37))
    with open(f'{args.log_dir}/test_metrics.txt','r') as f:
        for line in f.readlines():
            data = line.split(',')
            for i, d in enumerate(data[1:]):
                metrics[int(data[0])-1, i] = float(d)
    print('----------------------------------------------------------------')
    metrics_mean = np.mean(metrics, axis=0)
    metrics_std = np.std(metrics, axis=0)

    print(f'Results for {args.model} in {args.dataset_name}:')
    for i, metric in enumerate(metrics_list):
        print(f'{metric}: {metrics_mean[i]:.4f} ± {metrics_std[i]:.4f}')

    with open(os.path.join(args.log_dir, 'results.txt'), 'a') as f:
        f.write(f'{args.model}\n')
        for i, metric in enumerate(metrics_list):
            f.write(f'{metric}: {metrics_mean[i]:.4f} ± {metrics_std[i]:.4f}\n')
        f.write('\n')
    
    print(f'Evaluation on {args.dataset_name} for {args.model} is done!')
    print('----------------------------------------------------------------')