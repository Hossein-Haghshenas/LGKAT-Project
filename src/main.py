import argparse
import os
import sys
import numpy as np
from time import time
from data_loader import load_data
from train_n_runs import train_n_runs

np.random.seed(555)


parser = argparse.ArgumentParser()

'''
# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=15, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')

# NGCF
parser.add_argument('--layer_size', nargs='?', default='[32, 32]', help='Output sizes of every layer')
parser.add_argument('--adj_type', nargs='?', default='norm', help='Type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
parser.add_argument('--alg_type', nargs='?', default='gcmc', help='Type of the graph convolutional layer from {ngcf, gcn, gcmc}.')
parser.add_argument('--node_dropout_flag', type=int, default=0, help='0: Disable node dropout, 1: Activate node dropout')
parser.add_argument('--node_dropout', nargs='?', default='[0.1, 0.1]',
                    help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]',
                    help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

parser.add_argument('--model_type', nargs='?', default='KGCN+NGCF', help='{KGCN, KGCN+NGCF}.')
'''


# book
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='graphsage', help='which aggregator to use (sum, concat, neighbor, graphsage)')
parser.add_argument('--n_epochs', type=int, default=15, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=3, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--l2_weight', type=float, default=2e-5, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--cl_weight', type=float, default=0.01, help='weight of contrastive learning loss (start with 0.01, adjust if needed)')
parser.add_argument('--cl_temperature', type=float, default=1.0, help='temperature parameter for contrastive learning (1.0 is recommended)')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')

# NGCF
parser.add_argument('--layer_size', nargs='?', default='[64,64]', help='Output sizes of every layer')
parser.add_argument('--adj_type', nargs='?', default='norm', help='Type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
parser.add_argument('--alg_type', nargs='?', default='ngcf', help='Type of the graph convolutional layer from {ngcf, gcn, gcmc}.')
parser.add_argument('--node_dropout_flag', type=int, default=0, help='0: Disable node dropout, 1: Activate node dropout')
parser.add_argument('--node_dropout', nargs='?', default='[0.1, 0.1]',
                    help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]',
                    help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

parser.add_argument('--model_type', nargs='?', default='KGCN', help='{KGCN, KGCN+NGCF}.')



'''
# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=2, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')

# NGCF
parser.add_argument('--layer_size', nargs='?', default='[16, 16]', help='Output sizes of every layer')
parser.add_argument('--adj_type', nargs='?', default='norm', help='Type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
parser.add_argument('--alg_type', nargs='?', default='ngcf', help='Type of the graph convolutional layer from {ngcf, gcn, gcmc}.')
parser.add_argument('--node_dropout_flag', type=int, default=0, help='0: Disable node dropout, 1: Activate node dropout')
parser.add_argument('--node_dropout', nargs='?', default='[0.1, 0.1]',
                    help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]',
                    help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

parser.add_argument('--model_type', nargs='?', default='KGCN', help='{KGCN, KGCN+NGCF, KGCN+GCMC, KGCN+GCN}.')
parser.add_argument('--att', nargs='?', default='u_r', help='{avg, u_r, u_r_mlp, u_r_e}.')  # attention types
parser.add_argument('--smoothing_steps', type=int, default=1, help='k of A^k in gcn layer.')
# '''
parser.add_argument('--smoothing_steps', type=int, default=1, help='k of A^k in gcn layer.')
parser.add_argument('--runs', type=int, default=3, help='the number of epochs')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--att', nargs='?', default='u_r', help='{avg, u_r, u_r_mlp, u_r_e}.')  # attention types
'''
******************************************************************************
Methods for combining items embeddings from KGCN and CF (GCMC, NGCF or GCN)
'''
# gcn, graphsage and bi are three aggregation methods from KGAT
parser.add_argument('--agg_type', nargs='?', default='weighted_avg', help='{weighted_avg, gcn, graphsage, bi}.')  # attention types

# hyperparameter for weighted average
# alpha = 0: user embedding from CF, item embedding from KGCN
# alpha = 1: both user embedding and item embedding from CF
# alpha = 0.5, average of item embeddings from CF and KGCN
parser.add_argument('--alpha', type=float, default=0, \
                    help='hyperparameter for balancing item embeddings from KGCN and CF (GCMC, NGCF or GCN)')
'''
******************************************************************************
'''
parser.add_argument('--pretrain', type=int, default=0, help='0: no pretrain, 1: pretraining from kgcn.')
parser.add_argument('--logging', nargs='?', default='save', help='{save, ctr, topk}.')
parser.add_argument('--seed', type=int, default=555, help='random seed')

parser.add_argument(
    '--params', 
    nargs='?', 
    default='[\'dataset\',\'aggregator\',\'n_epochs\',\'neighbor_sample_size\','
            '\'dim\',\'n_iter\',\'batch_size\',\'l2_weight\',\'ls_weight\',\'lr\','
            '\'ratio\',\'layer_size\',\'adj_type\',\'alg_type\',\'node_dropout_flag\','
            '\'node_dropout\',\'mess_dropout\',\'model_type\',\'att\',\'agg_type\','
            '\'alpha\',\'smoothing_steps\',\'runs\',\'logging\',\'pretrain\','
            '\'cl_weight\',\'cl_temperature\']',
    help='paramerts')

# ---- MOKAR multi-objective optimization (NSGA-II) ----
parser.add_argument('--mode', type=str, default='train',
                    help="{train, optimize}; 'optimize' runs MOKAR NSGA-II over LGKAT")
parser.add_argument('--pop_size', type=int, default=20, help='MOKAR: NSGA-II population size')
parser.add_argument('--n_gen', type=int, default=15, help='MOKAR: number of generations')
parser.add_argument('--eval_epochs', type=int, default=5, help='MOKAR: training epochs per candidate')
parser.add_argument('--moo_seed', type=int, default=555, help='MOKAR: RNG seed')
parser.add_argument('--out_dir', type=str, default='../result', help='MOKAR: output directory')
parser.add_argument('--verbose', type=int, default=1, help='MOKAR: verbosity')

show_loss = False
show_time = False
show_topk = True
show_ctr = True

t = time()

'''
************************************************
fixed parameters
'''
args = parser.parse_args()

# MOKAR optimization mode: search configurations + KG relation subsets, then exit.
if getattr(args, 'mode', 'train') == 'optimize':
    from optimize_moo import run_optimization
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if getattr(args, 'batch_size', 0) in (0, None):
        args.batch_size = None  # let MOKAR pick the per-dataset default
    run_optimization(args)
    sys.exit(0)

data = load_data(args)

# train(args, data, show_loss, show_topk)
def train_case_study(args, data, show_loss, show_topk, show_ctr):
    pass


if args.logging == 'save':
    train_n_runs(args, data, show_loss, show_topk, show_ctr)
else:
    train_case_study(args, data, show_loss, show_topk, show_ctr)

if show_time:
    print('time used: %d s' % (time() - t))

'''
************************************************
'''
