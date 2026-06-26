"""
Plot performance-across-windows for the temporal protocol (OL-LGKAT Phase 3 acceptance:
"performance-across-windows plot produced").

Reads every result/temporal/<model>_<dataset>_<protocol>.csv, groups by (model, dataset),
and overlays the protocols (incremental vs batch) for AUC and NDCG@10 vs window index.
Saves one PNG per (model, dataset) into result/temporal/.

    conda run -n lgkat python scripts/plot_temporal.py
"""
import csv
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result', 'temporal')


def _read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def main():
    files = sorted(glob.glob(os.path.join(OUT_DIR, '*.csv')))
    if not files:
        print('no temporal CSVs found in %s' % OUT_DIR)
        return

    # group rows by (model, dataset) -> {protocol: rows}
    groups = defaultdict(dict)
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]   # model_dataset_protocol
        protocol = stem.split('_')[-1]
        key = stem[: -(len(protocol) + 1)]
        groups[key][protocol] = _read(path)

    for key, by_proto in groups.items():
        metrics = [('auc', 'AUC'), ('N@10', 'NDCG@10')]
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]
        for ax, (col, label) in zip(axes, metrics):
            for proto, rows in sorted(by_proto.items()):
                if rows and col in rows[0]:
                    xs = [int(r['window']) for r in rows]
                    ys = [float(r[col]) for r in rows]
                    ax.plot(xs, ys, marker='o', label=proto)
            ax.set_xlabel('evaluation window')
            ax.set_ylabel(label)
            ax.set_title('%s across time windows' % label)
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle(key)
        fig.tight_layout()
        out_png = os.path.join(OUT_DIR, key + '_curve.png')
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        print('wrote %s' % out_png)


if __name__ == '__main__':
    main()
