"""
OL-LGKAT Phase 7 — aggregate real result files into tables (CSV + LaTeX) and figures (PNG).

Reads ONLY from result/ (no hand-written numbers). Competitor numbers (GraphSAIL,
Light-Distillation, CKIN) are NOT fabricated: competitors_template.csv is emitted with our real
numbers filled and competitor cells left blank + a 'source' column to populate from the papers.

Sources consumed (each optional; skipped if absent):
  result/temporal/baselines_table.csv      -> baselines table
  result/phase6/ablation/summary.csv       -> ablation table + bar figure
  result/phase6/sens_*/summary.csv         -> sensitivity tables + figures
  result/temporal/*_coldstart.json         -> cold-start bucket table
  result/temporal/*.json (windows)         -> per-window NDCG@10 curve figure

Outputs to result/aggregate/. Run with a matplotlib-capable interpreter (system python works; the
lgkat env's matplotlib is ABI-broken):
    python scripts/aggregate_results.py
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(ROOT, 'result')
OUT = os.path.join(RES, 'aggregate')


def _read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _load_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (ValueError, OSError):
        print('  [skip unreadable json: %s]' % os.path.basename(path))
        return None


def write_table(name, headers, rows, caption):
    with open(os.path.join(OUT, name + '.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    align = 'l' + 'r' * (len(headers) - 1)
    lines = ['\\begin{table}[t]', '\\centering', '\\caption{%s}' % caption,
             '\\label{tab:%s}' % name, '\\begin{tabular}{%s}' % align, '\\toprule',
             ' & '.join(str(h).replace('_', '\\_') for h in headers) + ' \\\\', '\\midrule']
    for r in rows:
        lines.append(' & '.join(str(c).replace('_', '\\_') for c in r) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    with open(os.path.join(OUT, name + '.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('  table  %-22s (%d rows) -> .csv + .tex' % (name, len(rows)))


def agg_baselines():
    p = os.path.join(RES, 'temporal', 'baselines_table.csv')
    if not os.path.exists(p):
        return
    cols = ['method', 'avg_auc', 'avg_ndcg10', 'final_ndcg10', 'total_sec', 'peak_train_rows']
    rows = [[r.get(c, '') for c in cols] for r in _read_csv(p)]
    write_table('baselines', cols, rows, 'Continual-learning baselines under the temporal protocol.')


def _bar(labels, means, errs, ylabel, title, fname):
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(labels)), means, yerr=errs, capsize=4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=130)
        plt.close(fig)
        print('  figure %s' % fname)
    except Exception as e:
        print('  [figure %s skipped: %s]' % (fname, e))


def agg_ablation():
    p = os.path.join(RES, 'phase6', 'ablation', 'summary.csv')
    if not os.path.exists(p):
        return
    rows = _read_csv(p)
    cols = ['cell', 'avg_auc_mean', 'avg_auc_std', 'avg_ndcg10_mean', 'avg_ndcg10_std',
            'final_ndcg10_mean', 'final_ndcg10_std']
    write_table('ablation', cols, [[r.get(c, '') for c in cols] for r in rows],
                'Ablation: branches and fusion variants (mean$\\pm$std over seeds).')
    _bar([r['cell'] for r in rows], [float(r['avg_ndcg10_mean']) for r in rows],
         [float(r['avg_ndcg10_std']) for r in rows], 'avg NDCG@10', 'Ablation', 'fig_ablation.png')


def agg_sensitivity():
    for p in sorted(glob.glob(os.path.join(RES, 'phase6', 'sens_*', 'summary.csv'))):
        exp = os.path.basename(os.path.dirname(p))
        rows = _read_csv(p)
        cols = ['cell', 'avg_ndcg10_mean', 'avg_ndcg10_std', 'avg_auc_mean', 'avg_auc_std']
        write_table(exp, cols, [[r.get(c, '') for c in cols] for r in rows],
                    'Sensitivity: %s (mean$\\pm$std).' % exp.replace('_', ' '))
        _bar([r['cell'] for r in rows], [float(r['avg_ndcg10_mean']) for r in rows],
             [float(r['avg_ndcg10_std']) for r in rows], 'avg NDCG@10', exp, 'fig_%s.png' % exp)


def agg_coldstart():
    files = sorted(glob.glob(os.path.join(RES, 'temporal', '*_coldstart.json')))
    if not files:
        return
    data, buckets = {}, set()
    for p in files:
        method = os.path.basename(p).replace('_coldstart.json', '')
        d = _load_json(p)
        if not d or 'buckets' not in d:
            continue
        row = {}
        for b, v in d['buckets'].items():
            nd = v['ndcg'].get('10') if isinstance(v['ndcg'], dict) else None
            if nd is not None:
                row[b] = round(float(nd), 4)
                buckets.add(b)
        data[method] = row
    bnames = sorted(buckets, key=lambda s: int(s.split('-')[0].rstrip('+')))
    cols = ['method'] + bnames
    rows = [[m] + [data[m].get(b, '') for b in bnames] for m in sorted(data)]
    write_table('coldstart_ndcg10', cols, rows, 'Cold-start: NDCG@10 by user interaction-count bucket.')


def agg_curves():
    series = {}
    for p in sorted(glob.glob(os.path.join(RES, 'temporal', '*.json'))):
        if p.endswith('_coldstart.json') or p.endswith('_interpret.json'):
            continue
        d = _load_json(p)
        if not d:
            continue
        wins = d.get('windows', [])
        if not wins:
            continue
        kl = wins[0].get('k_list', [])
        if 10 not in kl:
            continue
        i = kl.index(10)
        series[os.path.splitext(os.path.basename(p))[0]] = (
            [w['window'] for w in wins], [w['ndcg'][i] for w in wins])
    if not series:
        return
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, (xs, ys) in sorted(series.items()):
            ax.plot(xs, ys, marker='o', label=label)
        ax.set_xlabel('window')
        ax.set_ylabel('NDCG@10')
        ax.set_title('Per-window NDCG@10')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'fig_curves.png'), dpi=130)
        plt.close(fig)
        print('  figure fig_curves.png')
    except Exception as e:
        print('  [curves figure skipped: %s]' % e)


def competitors_template():
    ours = []
    p = os.path.join(RES, 'temporal', 'baselines_table.csv')
    if os.path.exists(p):
        ours = [(r['method'], r.get('avg_ndcg10', ''), r.get('avg_auc', '')) for r in _read_csv(p)]
    path = os.path.join(OUT, 'competitors_template.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['method', 'dataset', 'NDCG@10', 'AUC', 'source'])
        for m, nd, auc in ours:
            w.writerow([m, 'ml1m', nd, auc, 'this work (result/)'])
        for comp in ('GraphSAIL', 'Light-Distillation', 'CKIN'):
            w.writerow([comp, 'ml1m', '', '', 'FILL FROM PUBLISHED PAPER - do not fabricate'])
    print('  competitors_template.csv (our rows filled; competitor cells blank)')


def main():
    os.makedirs(OUT, exist_ok=True)
    print('aggregating real result files -> %s' % os.path.relpath(OUT, ROOT))
    agg_baselines()
    agg_ablation()
    agg_sensitivity()
    agg_coldstart()
    agg_curves()
    competitors_template()
    print('done.')


if __name__ == '__main__':
    main()
