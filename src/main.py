import argparse
import numpy as np
from time import time
from data_loader import load_data
from train_n_runs import train_n_runs

np.random.seed(555)

parser = argparse.ArgumentParser()

# Book dataset configuration
parser.add_argument('--dataset', type=str, default='book',
                    help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='concat',
                    help='which aggregator to use (ignored if use_multihead=True)')
parser.add_argument('--n_epochs', type=int, default=20,
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
parser.add_argument('--alg_type', nargs='?', default='lightgcn',
                    help='Type of the graph convolutional layer from {ngcf, lightgcn}.')
parser.add_argument('--node_dropout_flag', type=int, default=1,
                    help='0: Disable node dropout, 1: Activate node dropout')
parser.add_argument('--node_dropout', nargs='?', default='[0.1, 0.1]',
                    help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]',
                    help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

parser.add_argument('--model_type', nargs='?', default='KGCN_LightGCN',
                    help='{KGCN, KGCN_NGCF, KGCN_LightGCN, NGCF, LightGCN}.')
parser.add_argument('--smoothing_steps', type=int,
                    default=1, help='k of A^k in gcn layer.')
parser.add_argument('--runs', type=int, default=3, help='the number of runs')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--att', nargs='?', default='h_ur_t',
                    help='Attention type {avg, u_r, u_r_mlp, h_ur_t, etc}.')

# Methods for combining items embeddings from KGCN and CF
parser.add_argument('--agg_type', nargs='?', default='weighted_avg',
                    help='{weighted_avg, gcn, graphsage, bi}.')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='hyperparameter for balancing item embeddings from KGCN and CF')

# NEW IMPROVED FEATURES
parser.add_argument('--use_multihead', type=int, default=1,
                    help='1: Use multi-head attention, 0: Use original aggregators')
parser.add_argument('--n_heads', type=int, default=4,
                    help='Number of attention heads (must divide dim evenly)')
parser.add_argument('--use_contrastive', type=int, default=1,
                    help='1: Enable contrastive learning, 0: Disable')
parser.add_argument('--ssl_temp', type=float, default=0.2,
                    help='Temperature for contrastive loss')
parser.add_argument('--ssl_weight', type=float, default=0.1,
                    help='Weight for contrastive loss')
parser.add_argument('--aug_dropout', type=float, default=0.1,
                    help='Dropout rate for data augmentation in contrastive learning')
parser.add_argument('--use_dynamic_alpha', type=int, default=1,
                    help='1: Learn alpha dynamically, 0: Use fixed alpha')
parser.add_argument('--early_stopping_patience', type=int,
                    default=5, help='Patience for early stopping')

parser.add_argument('--pretrain', type=int, default=0,
                    help='0: no pretrain, 1: pretraining from kgcn.')
parser.add_argument('--logging', nargs='?', default='save',
                    help='{save, ctr, topk}.')
parser.add_argument('--seed', type=int, default=555, help='random seed')

parser.add_argument(
    '--params',
    nargs='?',
    default='[\'dataset\',\'aggregator\',\'n_epochs\',\'neighbor_sample_size\','
            '\'dim\',\'n_iter\',\'batch_size\',\'l2_weight\',\'ls_weight\',\'lr\','
            '\'ratio\',\'layer_size\',\'adj_type\',\'alg_type\',\'node_dropout_flag\','
            '\'node_dropout\',\'mess_dropout\',\'model_type\',\'att\',\'agg_type\','
            '\'alpha\',\'smoothing_steps\',\'runs\',\'logging\',\'pretrain\','
            '\'use_multihead\',\'n_heads\',\'use_contrastive\',\'ssl_temp\',\'ssl_weight\','
            '\'use_dynamic_alpha\',\'early_stopping_patience\']',
    help='parameters')

show_loss = True
show_time = True
show_topk = True
show_ctr = True

t = time()

args = parser.parse_args()
data = load_data(args)


def train_case_study(args, data, show_loss, show_topk, show_ctr):
    pass


if args.logging == 'save':
    train_n_runs(args, data, show_loss, show_topk, show_ctr)
else:
    train_case_study(args, data, show_loss, show_topk, show_ctr)

if show_time:
    print('time used: %d s' % (time() - t))

print("\n" + "="*80)
print("IMPROVEMENTS IMPLEMENTED:")
print("="*80)
print("1. Multi-Head Attention: {} heads".format(args.n_heads if args.use_multihead else "Disabled"))
print("2. Contrastive Learning: {}".format(
    "Enabled" if args.use_contrastive else "Disabled"))
print("3. Dynamic Alpha Learning: {}".format(
    "Enabled" if args.use_dynamic_alpha else "Fixed alpha = {}".format(args.alpha)))
print("4. Layer Normalization: Enabled in aggregators")
print("5. Residual Connections: Enabled in multi-head attention")
print("6. Early Stopping Patience: {} epochs".format(
    args.early_stopping_patience))
print("="*80)
