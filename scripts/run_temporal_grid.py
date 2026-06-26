"""
Phase 6 grid runner: ablation + hyperparameter sensitivity over the temporal protocol.

Runs src/main.py (subprocess, args passed as a list so there are no shell-glob issues) for each
(cell x seed), copies each run's result JSON into result/phase6/<experiment>/, and aggregates
per-cell mean+-std over seeds into result/phase6/<experiment>/summary.{csv,md}.

Must be launched with the lgkat env (it reuses that interpreter for the child runs):
    conda run -n lgkat python scripts/run_temporal_grid.py --experiment ablation \
        --seeds 555,556,557 --dim 8 --windows 5 --epochs 1 --topk_users 30

Experiments: ablation | sens_dim | sens_lr | sens_window | sens_replay | sens_ewc
Heavy thesis runs (larger --dim/--epochs/--windows, --topk_users 0) are meant to be run by the
user (see docs/phase5_baselines.md style hand-off).
"""
import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src')
TEMPORAL_OUT = os.path.join(HERE, '..', 'result', 'temporal')


def base_config(a):
    dim = a.dim
    return {
        'dataset': 'ml1m', 'aggregator': 'concat', 'neighbor_sample_size': '4', 'dim': str(dim),
        'n_iter': '1', 'batch_size': str(a.batch_size), 'l2_weight': '1e-5', 'lr': str(a.lr),
        'layer_size': '[%d]' % dim, 'adj_type': 'norm', 'alg_type': 'lightgcn',
        'model_type': 'KGCN_LightGCN', 'node_dropout': '[0.1]', 'mess_dropout': '[0.1]',
        'node_dropout_flag': '1', 'alpha': '0.5', 'smoothing_steps': '1', 'pretrain': '0',
        'att': 'product', 'gpu_id': '-1', 'topk_user_num': str(a.topk_users),
        'topk_k_list': '[2,10,50,100]', 'split': 'temporal', 'n_windows': str(a.windows),
        'n_epochs': str(a.epochs), 'protocol': 'incremental', 'fusion': 'fixed',
        'baseline': 'none', 'replay_ratio': '1.0', 'ewc_lambda': '0.0', 'deterministic': '1',
    }


def experiments():
    return {
        # the decisive ablation: gate variants + remove each branch
        'ablation': [
            ('full_fixed', {'model_type': 'KGCN_LightGCN', 'fusion': 'fixed'}),
            ('full_gate', {'model_type': 'KGCN_LightGCN', 'fusion': 'gate'}),
            ('full_forgetting', {'model_type': 'KGCN_LightGCN', 'fusion': 'forgetting_gate'}),
            ('no_kg_lightgcn', {'model_type': 'LightGCN', 'fusion': 'fixed'}),
            ('no_lightgcn_kgcn', {'model_type': 'KGCN', 'fusion': 'fixed'}),
        ],
        'sens_dim': [('dim%d' % d, {'dim': str(d), 'layer_size': '[%d]' % d}) for d in (8, 16, 32)],
        'sens_lr': [('lr%s' % lr, {'lr': lr}) for lr in ('0.005', '0.02', '0.05')],
        'sens_window': [('T%d' % t, {'n_windows': str(t)}) for t in (5, 8, 10)],
        'sens_replay': [('replay%s' % r, {'fusion': 'fixed', 'baseline': 'replay', 'replay_ratio': r})
                        for r in ('0.5', '1.0', '2.0')],
        'sens_ewc': [('ewc%s' % l, {'fusion': 'fixed', 'baseline': 'ewc', 'ewc_lambda': l})
                     for l in ('1.0', '100.0', '10000.0')],
    }


def expected_tag(cfg):
    parts = [cfg['model_type'], cfg['dataset'], cfg['protocol'], cfg['fusion']]
    if cfg.get('baseline', 'none') != 'none':
        parts.append(cfg['baseline'])
    return '_'.join(parts)


def run_cell(cfg, seed):
    cmd = [sys.executable, 'main.py', '--seed', str(seed)]
    for k, v in cfg.items():
        cmd += ['--' + k, str(v)]
    subprocess.run(cmd, cwd=SRC, check=True)
    with open(os.path.join(TEMPORAL_OUT, expected_tag(cfg) + '.json'), encoding='utf-8') as f:
        return json.load(f)


def summarize(d):
    wins = d['windows']

    def nd10(w):
        kl = w['k_list']
        return w['ndcg'][kl.index(10)] if 10 in kl else float('nan')

    aucs = [w['auc'] for w in wins]
    nds = [nd10(w) for w in wins]
    return {'avg_auc': sum(aucs) / len(aucs), 'avg_ndcg10': sum(nds) / len(nds),
            'final_ndcg10': nds[-1]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', required=True)
    p.add_argument('--seeds', default='555,556,557')
    p.add_argument('--dim', type=int, default=8)
    p.add_argument('--windows', type=int, default=5)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--topk_users', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=16384)
    p.add_argument('--lr', default='2e-2')
    a = p.parse_args()

    seeds = [int(s) for s in a.seeds.split(',')]
    exps = experiments()
    if a.experiment not in exps:
        sys.exit('unknown experiment %r; choose from %s' % (a.experiment, list(exps)))
    cells = exps[a.experiment]

    outdir = os.path.join(HERE, '..', 'result', 'phase6', a.experiment)
    os.makedirs(outdir, exist_ok=True)

    summary = []
    for label, override in cells:
        per_seed = []
        for seed in seeds:
            cfg = dict(base_config(a))
            cfg.update(override)
            print('\n===== %s | seed %d =====' % (label, seed))
            d = run_cell(cfg, seed)
            shutil.copy(os.path.join(TEMPORAL_OUT, expected_tag(cfg) + '.json'),
                        os.path.join(outdir, '%s_seed%d.json' % (label, seed)))
            per_seed.append(summarize(d))

        row = {'cell': label, 'n_seeds': len(seeds)}
        for key in ('avg_auc', 'avg_ndcg10', 'final_ndcg10'):
            vals = [x[key] for x in per_seed]
            row[key + '_mean'] = round(statistics.mean(vals), 4)
            row[key + '_std'] = round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 4)
        summary.append(row)
        print('%-18s avg_ndcg10=%.4f+-%.4f' % (label, row['avg_ndcg10_mean'], row['avg_ndcg10_std']))

    cols = ['cell', 'n_seeds', 'avg_auc_mean', 'avg_auc_std', 'avg_ndcg10_mean', 'avg_ndcg10_std',
            'final_ndcg10_mean', 'final_ndcg10_std']
    with open(os.path.join(outdir, 'summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in summary:
            w.writerow(r)
    with open(os.path.join(outdir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('| ' + ' | '.join(cols) + ' |\n|' + '|'.join(['---'] * len(cols)) + '|\n')
        for r in summary:
            f.write('| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n')
    print('\nwrote %s/summary.{csv,md}' % outdir)


if __name__ == '__main__':
    main()
