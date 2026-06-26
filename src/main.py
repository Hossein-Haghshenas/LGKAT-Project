import argparse
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
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=15, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=3, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--l2_weight', type=float, default=2e-5, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
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

parser.add_argument('--topk_user_num', type=int, default=100,
                    help='Users for top-K eval; 0 = all users in train∩test (slow on large datasets)')
parser.add_argument('--topk_k_list', nargs='?', default='[1,2,5,10,20,50,100]',
                    help='K values for Recall@K, e.g. [2,10,50,100] to match paper tables')

# Phase 1 (OL-LGKAT): evaluation completeness. Both default ON so NDCG/structured output
# are produced by default; set both to 0 for byte-identical legacy (base-reproduction) logs.
parser.add_argument('--report_ndcg', type=int, default=1,
                    help='1: log NDCG@K alongside Precision/Recall (topk files + aggregate block); 0: legacy output')
parser.add_argument('--structured_out', type=int, default=1,
                    help='1: also write structured per-run results to result/structured/ (JSON + runs.csv); 0: off')

# Phase 3 (OL-LGKAT): temporal/streaming protocol. split='random' (default) keeps the base
# behaviour; split='temporal' windows the data by timestamp and runs the protocol below.
parser.add_argument('--split', nargs='?', default='random', help='{random, temporal} data split')
parser.add_argument('--n_windows', type=int, default=10, help='number of temporal windows (split=temporal)')
parser.add_argument('--protocol', nargs='?', default='incremental',
                    help='{incremental, batch} temporal training mode (split=temporal)')

# Phase 4 (OL-LGKAT): forgetting-aware fusion gate. fixed (base) | gate (adaptive per-item) |
# forgetting_gate (adaptive gate + per-item historical stability signal). Default 'fixed'.
parser.add_argument('--fusion', nargs='?', default='fixed',
                    help='{fixed, gate, forgetting_gate} item-embedding fusion')
parser.add_argument('--n_stability_feats', type=int, default=4,
                    help='per-item stability feature dim used by forgetting_gate')

# Phase 4: checkpoint/resume controls for the incremental temporal loop.
parser.add_argument('--resume', type=int, default=0, help='1: resume temporal run from saved state')
parser.add_argument('--stop_after_window', type=int, default=-1,
                    help='stop after this window index (for stop/resume testing); -1 = run all')
parser.add_argument('--deterministic', type=int, default=0,
                    help='1: single-thread + TF_DETERMINISTIC_OPS for reproducible resume')

# Phase 5: continual-learning baselines for the incremental loop (full-retrain = --protocol
# batch; fine-tune = incremental + fusion fixed + baseline none).
parser.add_argument('--baseline', nargs='?', default='none',
                    help='{none, replay, ewc} incremental CL baseline')
parser.add_argument('--replay_ratio', type=float, default=1.0,
                    help='experience replay: #replayed past samples per current-window sample')
parser.add_argument('--ewc_lambda', type=float, default=0.0,
                    help='EWC regularization strength (baseline ewc)')

# Phase 6: cold-start bucketed eval + gate interpretability (computed at the final window).
parser.add_argument('--coldstart', type=int, default=0,
                    help='1: also report Recall/NDCG bucketed by user interaction count')
parser.add_argument('--coldstart_buckets', nargs='?', default='[5,20,50]',
                    help='ascending interaction-count boundaries for cold-start buckets')
parser.add_argument('--interpret', type=int, default=0,
                    help='1: export the fusion gate case study for a sample user (gate fusions)')
parser.add_argument('--interpret_topk', type=int, default=10,
                    help='number of top items to export in the interpretability case study')

parser.add_argument(
    '--params',
    nargs='?', 
    default='[\'dataset\',\'aggregator\',\'n_epochs\',\'neighbor_sample_size\','
            '\'dim\',\'n_iter\',\'batch_size\',\'l2_weight\',\'ls_weight\',\'lr\','
            '\'ratio\',\'layer_size\',\'adj_type\',\'alg_type\',\'node_dropout_flag\','
            '\'node_dropout\',\'mess_dropout\',\'model_type\',\'att\',\'agg_type\','
            '\'alpha\',\'smoothing_steps\',\'runs\',\'logging\',\'pretrain\','
            '\'topk_user_num\',\'topk_k_list\',\'report_ndcg\',\'structured_out\']',
    help='paramerts')

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


# train(args, data, show_loss, show_topk)
def train_case_study(args, data, show_loss, show_topk, show_ctr):
    pass


if args.split == 'temporal':
    # Phase 3: temporal/streaming protocol (does not touch the base random-split flow).
    from data_loader import load_data_temporal
    from train_temporal import run_temporal_and_log
    data = load_data_temporal(args)
    run_temporal_and_log(args, data)
else:
    data = load_data(args)
    if args.logging == 'save':
        train_n_runs(args, data, show_loss, show_topk, show_ctr)
    else:
        train_case_study(args, data, show_loss, show_topk, show_ctr)

if show_time:
    print('time used: %d s' % (time() - t))

'''
************************************************
'''
