import argparse
import numpy as np
from time import time
from train_n_runs import train_n_runs

# NEW IMPORTS FOR DIFFUSION
from diffusion_integration import add_diffusion_arguments, load_data_with_diffusion

np.random.seed(555)

parser = argparse.ArgumentParser()

# ==================== ORIGINAL ARGUMENTS ====================
# book (default configuration)
parser.add_argument('--dataset', type=str, default='book',
                    help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum',
                    help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=15,
                    help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int,
                    default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=64,
                    help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=3,
                    help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--l2_weight', type=float, default=2e-5,
                    help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0,
                    help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--ratio', type=float, default=1,
                    help='size of training dataset')

# NGCF/LightGCN parameters
parser.add_argument('--layer_size', nargs='?',
                    default='[64,64]', help='Output sizes of every layer')
parser.add_argument('--adj_type', nargs='?', default='norm',
                    help='Type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
parser.add_argument('--alg_type', nargs='?', default='ngcf',
                    help='Type of the graph convolutional layer from {ngcf, gcn, gcmc}.')
parser.add_argument('--node_dropout_flag', type=int, default=0,
                    help='0: Disable node dropout, 1: Activate node dropout')
parser.add_argument('--node_dropout', nargs='?', default='[0.1, 0.1]',
                    help='Keep probability w.r.t.node dropout (i.e., 1-dropout_ratio) for each deep layer.1: no dropout.')
parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]',
                    help='Keep probability w.r.t.message dropout (i.e., 1-dropout_ratio) for each deep layer.1: no dropout.')

parser.add_argument('--model_type', nargs='?',
                    default='KGCN', help='{KGCN, KGCN+NGCF}.')

# Other parameters
parser.add_argument('--smoothing_steps', type=int,
                    default=1, help='k of A^k in gcn layer.')
parser.add_argument('--runs', type=int, default=3, help='the number of epochs')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--att', nargs='?', default='u_r',
                    help='{avg, u_r, u_r_mlp, u_r_e}.')

# Methods for combining items embeddings from KGCN and CF
parser.add_argument('--agg_type', nargs='?', default='weighted_avg',
                    help='{weighted_avg, gcn, graphsage, bi}.')
parser.add_argument('--alpha', type=float, default=0,
                    help='hyperparameter for balancing item embeddings from KGCN and CF (GCMC, NGCF or GCN)')

# Pretrain and logging
parser.add_argument('--pretrain', type=int, default=0,
                    help='0: no pretrain, 1: pretraining from kgcn.')
parser.add_argument('--logging', nargs='?', default='save',
                    help='{save, ctr, topk}.')
parser.add_argument('--seed', type=int, default=555, help='random seed')

# Adaptive Contrastive Learning parameter
parser.add_argument('--cl_weight', type=float, default=0.1,
                    help='weight of contrastive learning loss (0 = disabled, 0.05-0.15 recommended)')

# Parameters list
parser.add_argument(
    '--params',
    nargs='?',
    default='[\'dataset\',\'aggregator\',\'n_epochs\',\'neighbor_sample_size\','
            '\'dim\',\'n_iter\',\'batch_size\',\'l2_weight\',\'ls_weight\',\'lr\','
            '\'ratio\',\'layer_size\',\'adj_type\',\'alg_type\',\'node_dropout_flag\','
            '\'node_dropout\',\'mess_dropout\',\'model_type\',\'att\',\'agg_type\','
            '\'alpha\',\'smoothing_steps\',\'runs\',\'logging\',\'pretrain\',\'cl_weight\']',
    help='parameters')


# ==================== NEW: ADD DIFFUSION ARGUMENTS ====================
parser = add_diffusion_arguments(parser)


# ==================== PARSE ARGUMENTS ====================
show_loss = False
show_time = False
show_topk = True
show_ctr = True

t = time()

args = parser.parse_args()

# ==================== LOAD DATA (WITH DIFFUSION OPTION) ====================
# CHANGED: Use load_data_with_diffusion instead of load_data
data = load_data_with_diffusion(args)


# ==================== TRAINING ====================
def train_case_study(args, data, show_loss, show_topk, show_ctr):
    pass


if args.logging == 'save':
    train_n_runs(args, data, show_loss, show_topk, show_ctr)
else:
    train_case_study(args, data, show_loss, show_topk, show_ctr)

if show_time:
    print('time used: %d s' % (time() - t))
