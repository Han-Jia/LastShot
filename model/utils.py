import os
import shutil
import time
import pprint
import torch
import argparse
import numpy as np

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()    
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)

    return encoded_indicies

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def ensure_path(dir_path, scripts_to_save=None):
    # if os.path.exists(dir_path):
    #     if input('{} exists, remove? ([y]/n)'.format(dir_path)) != 'n':
    #         shutil.rmtree(dir_path)
    #         os.mkdir(dir_path)
    # else:
    #     os.mkdir(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for src_file in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(src_file))
            print('copy {} to {}'.format(src_file, dst_file))
            if os.path.isdir(src_file):
                shutil.copytree(src_file, dst_file)
            else:
                shutil.copyfile(src_file, dst_file)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def postprocess_args(args):
    save_path1 = '-'.join([args.dataset, args.model_class, args.backbone_class, args.backbone_class_t, '{:02d}w{:02d}s{:02}q'.format(args.way, args.shot, args.query)])
    if args.use_repo > 0:
        save_path1 += '-Repo{}-{}'.format(args.use_repo, args.kd_type)
        save_path2 = '_'.join([str('_'.join(args.step_size.split(','))), str(args.gamma),
                               'lr{:.2g}'.format(args.lr),
                               str(args.lr_scheduler), 
                               'kd{}kdt{}'.format(args.kd_temperature, args.temperature),
                               'b{}'.format(args.balance),
                               'bsz{:03d}'.format( args.way*(args.shot+args.query) ),
                               # str(time.strftime('%Y%m%d_%H%M%S'))
                               ])   
    else:
        save_path2 = '_'.join([str('_'.join(args.step_size.split(','))), str(args.gamma),
                               'lr{:.2g}'.format(args.lr),
                               str(args.lr_scheduler), str(args.temperature), 
                               'b{}'.format(args.balance),
                               'bsz{:03d}'.format(args.way*(args.shot+args.query) ),
                               # str(time.strftime('%Y%m%d_%H%M%S'))
                               ])   
    if args.para_init is not None:
        save_path1 += '-Pre'  
    if args.model_class in ['FEAT']:
        save_path2 += '-Mul{}BF{}dp{}h{}l{}T2{}'.format(args.lr_mul, args.balance_feat, args.dp_rate, args.n_head, args.n_layer, args.temperature)
    if args.fix_BN:
        save_path2 += '-FBN'
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, save_path1)):
        os.mkdir(os.path.join(args.save_dir, save_path1))
    args.save_path = os.path.join(args.save_dir, save_path1, save_path2)
    return args

def get_command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=500)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--eval_shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--eval_query', type=int, default=15)
    parser.add_argument('--model_class', type=str, default='FEAT', choices=['ProtoNet', 'FEAT'])
    parser.add_argument('--backbone_class', type=str, default='Res12',
                        choices=['ConvNet', 'Res12'])
    parser.add_argument('--backbone_class_t', type=str, default='Res12',
                        choices=['ConvNet', 'Res12'])    
    parser.add_argument('--dataset', type=str, default='MiniImageNet', 
                        choices=['MiniImageNet', 'TieredImageNet']) # 'CUB', 'CIFARFS', 'FC100' will be released in future
 
    # optimization parameters
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_mul', type=float, default=1)    
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='100')
    parser.add_argument('--gamma', type=float, default=0.5)    
    parser.add_argument('--fix_BN', action='store_true', default=False)    
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', default='0')
    
    # frequent parameters
    parser.add_argument('--orig_imsize', type=int, default=-1)                              # -1 for no cache, and -2 for no resize
    parser.add_argument('--balance', type=float, default=0)
    parser.add_argument('--use_euclidean', action='store_true', default=False)
    parser.add_argument('--temperature', type=float, default=64)
    parser.add_argument('--kd_temperature', type=float, default=1)  
    parser.add_argument('--kd_type', type=str, default='NN', choices=['NN', 'LR', 'SVM'])   # use which model to distill
    parser.add_argument('--kd_c', type=float, default=1)
    parser.add_argument('--para_init', type=str, default='./saves/initialization/miniimagenet/Res12-pre.pth')                           # ConvNet: './saves/initialization/miniimagenet/con_pre.pth'
    parser.add_argument('--repo_init', type=str, default='./saves/initialization/miniimagenet/Res12-pre.pth')                              # ConvNet: './saves/initialization/miniimagenet/con_pre.pth'    
    parser.add_argument('--use_repo', type=int, default=1)                                  # after the iteration, we use meta-repo, if 0, we do not use it     
    
    # for FEAT-based models
    parser.add_argument('--dp_rate', default=0.5, type=float, help='dropout for FEAT')    
    parser.add_argument('--n_head', default=1, type=int, help='number of heads for FEAT')    
    parser.add_argument('--n_layer', default=1, type=int, help='number of layers for FEAT')    
    parser.add_argument('--temperature2', type=float, default=1)                            # only for FEAT    
    parser.add_argument('--balance_feat', type=float, default=0)    

    # usually untouched parameters
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    return parser