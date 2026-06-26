"""
Build the Phase 5 "quality vs time / memory" comparison table for the temporal baselines.

Reads every result/temporal/<...>.json (one per run: Full-Retrain / Fine-Tune / Replay / EWC /
OL-LGKAT gate) and, per method, reports avg AUC, avg & final NDCG@10, total wall-clock, avg
sec/window, and peak training rows per window (a compute/memory proxy: Full-Retrain grows with
history, Replay ~2x, Fine-Tune/gate ~1x; EWC additionally stores Fisher+anchor params).

Writes result/temporal/baselines_table.{csv,md}.

    conda run -n lgkat python scripts/baselines_table.py
"""
import csv
import glob
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result', 'temporal')


def method_label(d):
    protocol = d.get('protocol')
    fusion = d.get('fusion', 'fixed')
    baseline = d.get('baseline', 'none')
    if protocol == 'batch':
        return 'Full-Retrain (batch)'
    if fusion == 'forgetting_gate':
        return 'OL-LGKAT (forgetting gate)'
    if fusion == 'gate':
        return 'Adaptive gate'
    if baseline == 'replay':
        return 'Experience Replay'
    if baseline == 'ewc':
        return 'EWC'
    return 'Fine-Tune'


def _ndcg_at(w, k=10):
    kl = w.get('k_list', [])
    return float(w['ndcg'][kl.index(k)]) if k in kl else float('nan')


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT, '*.json'))):
        d = json.load(open(path, encoding='utf-8'))
        wins = d.get('windows', [])
        if not wins:
            continue
        aucs = [w['auc'] for w in wins]
        ndcgs = [_ndcg_at(w, 10) for w in wins]
        secs = [w.get('seconds', 0.0) for w in wins]
        trows = [w.get('train_rows', 0) for w in wins]
        rows.append({
            'method': method_label(d), 'dataset': d.get('dataset'),
            'protocol': d.get('protocol'), 'fusion': d.get('fusion'),
            'baseline': d.get('baseline', 'none'),
            'avg_auc': round(sum(aucs) / len(aucs), 4),
            'avg_ndcg10': round(sum(ndcgs) / len(ndcgs), 4),
            'final_ndcg10': round(ndcgs[-1], 4),
            'total_sec': round(sum(secs), 1),
            'avg_sec_per_window': round(sum(secs) / len(secs), 1),
            'peak_train_rows': max(trows), 'n_windows_eval': len(wins),
        })
    if not rows:
        print('no temporal result JSONs found in %s' % OUT)
        return

    cols = ['method', 'dataset', 'protocol', 'fusion', 'baseline', 'avg_auc', 'avg_ndcg10',
            'final_ndcg10', 'total_sec', 'avg_sec_per_window', 'peak_train_rows', 'n_windows_eval']
    rows.sort(key=lambda x: x['avg_ndcg10'], reverse=True)

    with open(os.path.join(OUT, 'baselines_table.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(os.path.join(OUT, 'baselines_table.md'), 'w', encoding='utf-8') as f:
        f.write('| ' + ' | '.join(cols) + ' |\n')
        f.write('|' + '|'.join(['---'] * len(cols)) + '|\n')
        for r in rows:
            f.write('| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n')

    for r in rows:
        print('%-28s avgAUC=%.4f avgNDCG@10=%.4f total=%.1fs peakRows=%d'
              % (r['method'], r['avg_auc'], r['avg_ndcg10'], r['total_sec'], r['peak_train_rows']))
    print('\nwrote baselines_table.csv / .md to %s' % OUT)


if __name__ == '__main__':
    main()
