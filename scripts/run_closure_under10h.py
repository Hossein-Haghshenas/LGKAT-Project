# sequential under-10h coding-closure campaign (tasks 1, 2, 4, 6).
# skips tasks 3 and 5 (all-users / sparsity, each >10h).
# run from repo root:
#   conda run -n lgkat --no-capture-output python scripts/run_closure_under10h.py
# one stage:
#   conda run -n lgkat --no-capture-output python scripts/run_closure_under10h.py --only 1
from __future__ import print_function

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
RESULT = os.path.join(ROOT, 'result')
NOTES = os.path.join(ROOT, 'docs', 'THESIS_NOTES')
PY = sys.executable
K_LIST = '[2,10,50,100]'

# paper Table 2/3 Product LGKAT (scripts/paper_targets_product.py)
PAPER_R10 = {'movie': 0.1966, 'book': 0.0696, 'music': 0.1407, 'restaurant': 0.1406}
PAPER_F1 = {'movie': 0.9375, 'book': 0.6495, 'music': 0.7418, 'restaurant': 0.7816}


def _env():
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONHASHSEED'] = '0'
    return env


def _cmd(pairs):
    out = [PY, 'main.py']
    for key, val in pairs:
        out.extend(['--' + key, str(val)])
    return out


def music_ssl():
    return [
        ('dataset', 'music'), ('aggregator', 'concat'), ('n_epochs', '39'),
        ('neighbor_sample_size', '8'), ('dim', '32'), ('n_iter', '1'),
        ('batch_size', '128'), ('l2_weight', '1e-4'), ('lr', '0.005'),
        ('layer_size', '[32]'), ('adj_type', 'norm'), ('alg_type', 'lightgcn'),
        ('model_type', 'KGCN_LightGCN'), ('node_dropout', '[0.1]'),
        ('mess_dropout', '[0.1]'), ('node_dropout_flag', '1'), ('alpha', '0.5'),
        ('smoothing_steps', '8'), ('pretrain', '0'), ('att', 'product'),
        ('gpu_id', '0'), ('topk_k_list', K_LIST), ('topk_user_num', '100'),
        ('fusion', 'gate'), ('use_cl', '1'), ('cl_lambda', '0.02'),
        ('cl_tau', '0.2'), ('use_simgcl', '1'), ('simgcl_lambda', '0.02'),
        ('simgcl_eps', '0.05'), ('lr_decay', '0.98'), ('early_stop', '12'),
        ('fast_eval', '1'), ('log_resources', '1'),
    ]


def book_champ():
    return [
        ('dataset', 'book'), ('aggregator', 'concat'), ('n_epochs', '39'),
        ('neighbor_sample_size', '8'), ('dim', '64'), ('n_iter', '1'),
        ('batch_size', '256'), ('l2_weight', '2e-5'), ('lr', '0.0002'),
        ('layer_size', '[64]'), ('adj_type', 'norm'), ('alg_type', 'lightgcn'),
        ('model_type', 'KGCN_LightGCN'), ('node_dropout', '[0.1]'),
        ('mess_dropout', '[0.1]'), ('node_dropout_flag', '1'), ('alpha', '0'),
        ('smoothing_steps', '3'), ('pretrain', '0'), ('att', 'product'),
        ('gpu_id', '0'), ('topk_k_list', K_LIST), ('topk_user_num', '100'),
        ('fusion', 'gate'), ('use_cl', '0'), ('use_simgcl', '0'),
        ('lr_decay', '0.98'), ('early_stop', '5'), ('early_stop_min_epoch', '0'),
        ('fast_eval', '1'), ('log_resources', '1'),
    ]


def movie_champ():
    return [
        ('dataset', 'movie'), ('aggregator', 'concat'), ('n_epochs', '39'),
        ('neighbor_sample_size', '4'), ('dim', '32'), ('n_iter', '2'),
        ('batch_size', '8192'), ('l2_weight', '5e-6'), ('lr', '0.0025'),
        ('layer_size', '[32]'), ('adj_type', 'plain'), ('alg_type', 'lightgcn'),
        ('model_type', 'KGCN_LightGCN'), ('node_dropout', '[0.1]'),
        ('mess_dropout', '[0.1]'), ('node_dropout_flag', '1'), ('alpha', '0'),
        ('smoothing_steps', '1'), ('pretrain', '0'), ('att', 'product'),
        ('gpu_id', '0'), ('topk_k_list', K_LIST), ('topk_user_num', '100'),
        ('fusion', 'fixed'), ('use_cl', '0'), ('use_simgcl', '0'),
        ('lr_decay', '0.98'), ('early_stop', '8'), ('early_stop_min_epoch', '0'),
        ('fast_eval', '1'), ('log_resources', '1'),
    ]


def restaurant_champ():
    return [
        ('dataset', 'restaurant'), ('aggregator', 'concat'), ('n_epochs', '39'),
        ('neighbor_sample_size', '4'), ('dim', '8'), ('n_iter', '2'),
        ('batch_size', '32768'), ('l2_weight', '1e-7'), ('lr', '0.02'),
        ('layer_size', '[8]'), ('adj_type', 'norm'), ('alg_type', 'lightgcn'),
        ('model_type', 'KGCN_LightGCN'), ('node_dropout', '[0.1]'),
        ('mess_dropout', '[0.1]'), ('node_dropout_flag', '1'),
        ('agg_type', 'weighted_avg'), ('smoothing_steps', '1'), ('pretrain', '0'),
        ('alpha', '0.5'), ('att', 'product'), ('gpu_id', '0'),
        ('topk_k_list', K_LIST), ('topk_user_num', '100'), ('fusion', 'fixed'),
        ('use_cl', '0'), ('use_simgcl', '0'), ('lr_decay', '0.98'),
        ('early_stop', '5'), ('early_stop_min_epoch', '0'),
        ('fast_eval', '1'), ('log_resources', '1'),
    ]


def overlay(base, **kw):
    d = dict(base)
    for k, v in kw.items():
        d[k] = v
    return list(d.items())


def runs_json(dataset, att, tag):
    return os.path.join(RESULT, 'runs_KGCN_LightGCN_{}_{}_{}.json'.format(
        dataset, att, tag))


def interpret_json(dataset, att, tag):
    return os.path.join(RESULT, 'KGCN_LightGCN_{}_attention_{}_{}.json'.format(
        dataset, att, tag))


def runs_done(dataset, att, tag, min_runs):
    path = runs_json(dataset, att, tag)
    if not os.path.exists(path):
        return False
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    n_ok = int(data.get('n_valid') or data.get('runs') or 0)
    return n_ok >= int(min_runs)


def load_runs(dataset, att, tag):
    path = runs_json(dataset, att, tag)
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def fmt_ms(vals):
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 'n/a'
    return '{:.4f} ± {:.4f} (n={})'.format(float(arr.mean()), float(arr.std()), int(arr.size))


def recall_at(data, k):
    rec = np.array(data.get('recall') or [], dtype=float)
    k_list = list(data.get('k_list') or [])
    if rec.size == 0 or k not in k_list:
        return None
    return rec[:, k_list.index(k)]


def mean_metric(data, name, k=None):
    if name in ('auc', 'f1'):
        return fmt_ms(data.get(name) or [])
    arr = recall_at(data, k) if name == 'recall' else None
    if name == 'ndcg':
        rec = np.array(data.get('ndcg') or [], dtype=float)
        k_list = list(data.get('k_list') or [])
        if rec.size == 0 or k not in k_list:
            return 'n/a'
        arr = rec[:, k_list.index(k)]
    if arr is None:
        return 'n/a'
    return fmt_ms(arr)


def run_arm(name, cmd, skip):
    print('=' * 80, flush=True)
    print('ARM %s' % name, flush=True)
    if skip:
        print('SKIP already complete: %s' % name, flush=True)
        print('ARM_DONE %s' % name, flush=True)
        return 0
    print(' '.join(cmd), flush=True)
    print('=' * 80, flush=True)
    rc = subprocess.call(cmd, env=_env(), cwd=SRC)
    if rc != 0:
        print('ARM_FAILED %s rc=%s' % (name, rc), flush=True)
        return rc
    print('ARM_DONE %s' % name, flush=True)
    return 0


def git_save(message, paths):
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        return
    subprocess.call(['git', 'add', '--'] + existing, cwd=ROOT)
    rc = subprocess.call(['git', 'status', '--porcelain'], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # commit even if porcelain check is coarse; empty commit fails and is fine
    subprocess.call(['git', 'commit', '-m', message], cwd=ROOT)


def sig_test(dataset, proposed_tag, base_tag, out_name):
    cmd = [
        PY, os.path.join(ROOT, 'scripts', 'significance_test.py'),
        '--dataset', dataset,
        '--proposed_model', 'KGCN_LightGCN',
        '--base_model', 'KGCN_LightGCN',
        '--proposed_att', 'product',
        '--base_att', 'product',
        '--proposed_tag', proposed_tag,
        '--base_tag', base_tag,
        '--k', '10',
    ]
    print('SIG', ' '.join(cmd), flush=True)
    src = os.path.join(RESULT, 'significance_{}.txt'.format(dataset))
    dst = os.path.join(RESULT, out_name)
    bak = src + '.bak_closure'
    if os.path.exists(src):
        shutil.copyfile(src, bak)
    subprocess.call(cmd, cwd=ROOT)
    text = ''
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        with open(src, encoding='utf-8') as f:
            text = f.read()
        print('copied significance -> %s' % dst, flush=True)
    if os.path.exists(bak):
        shutil.copyfile(bak, src)
        os.remove(bak)
    return text


def interpret_row(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    n_nb = int(data.get('n_neighbor') or 0)
    recs = data.get('recommendations') or []
    n_unique = []
    att_stds = []
    n_repeat_slots = 0
    n_slots = 0
    examples = []
    for rec in recs:
        nbs = rec.get('kg_neighbors') or []
        ents = [int(x['entity']) for x in nbs]
        atts = [float(x['attention']) for x in nbs]
        uniq = sorted(set(ents))
        n_unique.append(len(uniq))
        if atts:
            att_stds.append(float(np.std(atts)))
        n_slots += len(ents)
        n_repeat_slots += len(ents) - len(uniq)
        top = sorted(nbs, key=lambda d: -float(d['attention']))[:3]
        examples.append({
            'rank': rec.get('rank'),
            'item': rec.get('item'),
            'in_test': rec.get('in_test'),
            'n_unique_entities': len(uniq),
            'att_std': float(np.std(atts)) if atts else None,
            'top3': [{'entity': int(t['entity']), 'relation': int(t['relation']),
                      'attention': float(t['attention'])} for t in top],
        })
    uniform = bool(att_stds) and (float(np.mean(att_stds)) < 1e-3)
    return {
        'dataset': data.get('dataset'),
        'user': data.get('user'),
        'n_neighbor': n_nb,
        'n_recs': len(recs),
        'mean_unique_entities': float(np.mean(n_unique)) if n_unique else 0.0,
        'repeat_slot_frac': (n_repeat_slots / float(n_slots)) if n_slots else 0.0,
        'mean_att_std': float(np.mean(att_stds)) if att_stds else 0.0,
        'weights_uniform': uniform,
        'examples': examples,
    }


def write_note(name, body):
    os.makedirs(NOTES, exist_ok=True)
    path = os.path.join(NOTES, name)
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    text = u'# %s\n\nتاریخ استخراج: %s\n\n%s\n' % (name.replace('.md', ''), stamp, body)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print('WROTE %s' % path, flush=True)
    return path


def summarize_interpret():
    lines = [
        'خروجی تفسیرپذیری قهرمان هر مجموعه‌داده (`--interpret 1 --exp_tag interp_v2`).',
        'اعداد از فایل‌های `result/KGCN_LightGCN_*_attention_product_interp_v2.json` خوانده شده‌اند.',
        '',
        '| مجموعه‌داده | کاربر | همسایه | میانگین موجودیت یکتا | سهم جایگاه تکراری | انحراف‌معیار وزن | وزن یکنواخت؟ |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    extra = []
    for ds in ('music', 'book', 'movie', 'restaurant'):
        path = interpret_json(ds, 'product', 'interp_v2')
        if not os.path.exists(path):
            lines.append('| %s | — | — | — | — | — | فایل نیست |' % ds)
            continue
        row = interpret_row(path)
        lines.append(
            '| {dataset} | {user} | {n_neighbor} | {mean_unique_entities:.2f} | '
            '{repeat_slot_frac:.2f} | {mean_att_std:.4f} | {uni} |'.format(
                uni='بله' if row['weights_uniform'] else 'خیر', **row))
        extra.append('## %s — کاربر %s' % (ds, row['user']))
        extra.append('')
        for ex in row['examples']:
            top = ', '.join(
                'e=%s r=%s a=%.4f' % (t['entity'], t['relation'], t['attention'])
                for t in ex['top3'])
            extra.append(
                '- رتبه %s آیتم %s (in_test=%s) موجودیت یکتا=%s att_std=%.4f؛ top3: %s' % (
                    ex['rank'], ex['item'], ex['in_test'],
                    ex['n_unique_entities'], ex['att_std'] or 0.0, top))
        extra.append('')
    body = '\n'.join(lines + [''] + extra)
    body += (
        '\n**برای بخش ۴-۷:** اگر لست‌اف‌ام و بوک کراسینگ وزن نزدیک ۱/n و موجودیت تکراری داشته باشند، '
        'و مووی‌لنز/دیان‌پینگ وزن نایکنواخت و موجودیت متمایز، ادعا از استدلال ساختاری به شاهد تجربی می‌رسد.\n'
    )
    return write_note('01_interpret.md', body)


def summarize_screens():
    lines = [
        'غربالگری توجه حاصل‌ضربی (`--att product`)؛ پروتکل ۱۰۰ کاربر، ۵ اجرا.',
        '',
        '## بوک کراسینگ — همجوشی fixed / gate / residual',
        '',
        '| بازو | AUC | F1 | Recall@10 | NDCG@10 |',
        '|---|---|---|---|---|',
    ]
    for tag, label in (
        ('v3_book_res_fixed_product', 'fixed'),
        ('v3_book_res_gate_product', 'gate'),
        ('v3_book_res_residual_product', 'residual'),
    ):
        data = load_runs('book', 'product', tag)
        if data is None:
            lines.append('| %s | — | — | — | — |' % label)
            continue
        lines.append('| %s | %s | %s | %s | %s |' % (
            label, mean_metric(data, 'auc'), mean_metric(data, 'f1'),
            mean_metric(data, 'recall', 10), mean_metric(data, 'ndcg', 10)))
    sig_book = sig_test(
        'book', 'v3_book_res_residual_product', 'v3_book_res_fixed_product',
        'significance_v3_book_res_product.txt')
    lines.extend(['', '### آزمون معناداری residual در برابر fixed', '', '```', sig_book.strip(), '```', ''])
    lines.extend([
        '## لست‌اف‌ام — hop1 در برابر hop2',
        '',
        '| بازو | AUC | F1 | Recall@10 | NDCG@10 |',
        '|---|---|---|---|---|',
    ])
    for tag, label in (
        ('v3_music_hop1_product', 'n_iter=1'),
        ('v3_music_hop2_product', 'n_iter=2'),
    ):
        data = load_runs('music', 'product', tag)
        if data is None:
            lines.append('| %s | — | — | — | — |' % label)
            continue
        lines.append('| %s | %s | %s | %s | %s |' % (
            label, mean_metric(data, 'auc'), mean_metric(data, 'f1'),
            mean_metric(data, 'recall', 10), mean_metric(data, 'ndcg', 10)))
    sig_music = sig_test(
        'music', 'v3_music_hop2_product', 'v3_music_hop1_product',
        'significance_v3_music_hop_product.txt')
    lines.extend(['', '### آزمون معناداری hop2 در برابر hop1', '', '```', sig_music.strip(), '```', ''])
    return write_note('02_product_screens.md', '\n'.join(lines))


def summarize_ssl():
    lines = [
        'حذف اجزای یادگیری خودنظارتی با سه اجرا. بازوی بدون CL در برابر بازوی با CL.',
        'توجه: برچسب قدیمی `..._005` یعنی `cl_lambda=0.005` نه ۰٫۰۵.',
        '',
        '| مجموعه‌داده | بازو | AUC | F1 | Recall@10 |',
        '|---|---|---|---|---|',
    ]
    specs = (
        ('book', 'ssl_ablate_3run_book_nocl', 'بدون CL'),
        ('book', 'ssl_ablate_3run_book_cl005', 'CL λ=0.005'),
        ('movie', 'ssl_ablate_3run_movie_nocl', 'بدون CL'),
        ('movie', 'ssl_ablate_3run_movie_cl005', 'CL λ=0.005'),
        ('restaurant', 'ssl_ablate_3run_restaurant_nocl', 'بدون CL'),
        ('restaurant', 'ssl_ablate_3run_restaurant_cl256', 'CL λ=0.02 sample=256'),
    )
    for ds, tag, label in specs:
        data = load_runs(ds, 'product', tag)
        if data is None:
            lines.append('| %s | %s | — | — | — |' % (ds, label))
            continue
        lines.append('| %s | %s | %s | %s | %s |' % (
            ds, label, mean_metric(data, 'auc'), mean_metric(data, 'f1'),
            mean_metric(data, 'recall', 10)))
    pairs = (
        ('book', 'ssl_ablate_3run_book_cl005', 'ssl_ablate_3run_book_nocl',
         'significance_ssl_ablate_book.txt'),
        ('movie', 'ssl_ablate_3run_movie_cl005', 'ssl_ablate_3run_movie_nocl',
         'significance_ssl_ablate_movie.txt'),
        ('restaurant', 'ssl_ablate_3run_restaurant_cl256', 'ssl_ablate_3run_restaurant_nocl',
         'significance_ssl_ablate_restaurant.txt'),
    )
    for ds, prop, base, out_name in pairs:
        table = sig_test(ds, prop, base, out_name)
        lines.extend(['', '## معناداری %s (CL در برابر بدون CL)' % ds, '', '```', table.strip(), '```'])
    lines.append(
        '\nاگر دلتا منفی یا ns باشد، همان پاسخ روشن به «چرا یادگیری تقابلی فقط روی لست‌اف‌ام روشن است» است.\n'
    )
    return write_note('04_ssl_ablation.md', '\n'.join(lines))


def summarize_paper():
    lines = [
        'بازتولید خط‌مبنای مقاله با خاموش‌کردن بهبودها: '
        '`fusion=fixed use_cl=0 use_simgcl=0 lr_decay=1.0 early_stop=0 runs=3`.',
        'موسیقی از قبل موجود است (`thesis_paper_product_sanity`).',
        '',
        '| مجموعه‌داده | F1 ما | F1 مقاله | Recall@10 ما | Recall@10 مقاله |',
        '|---|---|---|---|---|',
    ]
    for ds in ('music', 'book', 'movie', 'restaurant'):
        data = load_runs(ds, 'product', 'thesis_paper_product_sanity')
        if data is None:
            lines.append('| %s | — | %.4f | — | %.4f |' % (ds, PAPER_F1[ds], PAPER_R10[ds]))
            continue
        f1_vals = np.array(data.get('f1') or [], dtype=float)
        r10 = recall_at(data, 10)
        f1_s = fmt_ms(f1_vals)
        r_s = fmt_ms(r10) if r10 is not None else 'n/a'
        lines.append('| %s | %s | %.4f | %s | %.4f |' % (
            ds, f1_s, PAPER_F1[ds], r_s, PAPER_R10[ds]))
    lines.append(
        '\nاین اعداد جدول‌های ۴-۹ تا ۴-۱۱ را از نقل‌قول مقاله به بازاجرا زیر پروتکل یکسان ارتقا می‌دهند.\n'
    )
    return write_note('06_paper_baseline.md', '\n'.join(lines))


def stage_interpret(commit):
    arms = (
        ('interp_music', overlay(music_ssl(), runs='1', interpret='1',
                                 interpret_topk='5', exp_tag='interp_v2'),
         os.path.exists(interpret_json('music', 'product', 'interp_v2'))),
        ('interp_book', overlay(book_champ(), runs='1', interpret='1',
                                interpret_topk='5', exp_tag='interp_v2'),
         os.path.exists(interpret_json('book', 'product', 'interp_v2'))),
        ('interp_movie', overlay(movie_champ(), runs='1', interpret='1',
                                 interpret_topk='5', exp_tag='interp_v2'),
         os.path.exists(interpret_json('movie', 'product', 'interp_v2'))),
        ('interp_restaurant', overlay(restaurant_champ(), runs='1', interpret='1',
                                      interpret_topk='5', exp_tag='interp_v2'),
         os.path.exists(interpret_json('restaurant', 'product', 'interp_v2'))),
    )
    for name, pairs, skip in arms:
        rc = run_arm(name, _cmd(pairs), skip)
        if rc != 0:
            return rc
    note = summarize_interpret()
    if commit:
        git_save('feat: interpretability case study on four datasets',
                 [note, RESULT, os.path.join(ROOT, 'scripts', 'run_closure_under10h.py')])
    print('STAGE_DONE 1 interpret', flush=True)
    return 0


def stage_screens(commit):
    book_base = overlay(
        book_champ(), runs='5', rank_ckpt='r10', fusion='fixed')
    arms = [
        ('v3_book_res_fixed_product',
         overlay(book_base, fusion='fixed', exp_tag='v3_book_res_fixed_product'),
         runs_done('book', 'product', 'v3_book_res_fixed_product', 5)),
        ('v3_book_res_gate_product',
         overlay(book_base, fusion='gate', exp_tag='v3_book_res_gate_product'),
         runs_done('book', 'product', 'v3_book_res_gate_product', 5)),
        ('v3_book_res_residual_product',
         overlay(book_base, fusion='residual', exp_tag='v3_book_res_residual_product'),
         runs_done('book', 'product', 'v3_book_res_residual_product', 5)),
    ]
    music_base = overlay(
        music_ssl(), runs='5', rank_ckpt='r10', early_stop_min_epoch='25')
    arms += [
        ('v3_music_hop1_product',
         overlay(music_base, n_iter='1', exp_tag='v3_music_hop1_product'),
         runs_done('music', 'product', 'v3_music_hop1_product', 5)),
        ('v3_music_hop2_product',
         overlay(music_base, n_iter='2', exp_tag='v3_music_hop2_product'),
         runs_done('music', 'product', 'v3_music_hop2_product', 5)),
    ]
    for name, pairs, skip in arms:
        rc = run_arm(name, _cmd(pairs), skip)
        if rc != 0:
            return rc
    note = summarize_screens()
    if commit:
        git_save('feat: product-attention residual and hop screens',
                 [note, RESULT])
    print('STAGE_DONE 2 product-screens', flush=True)
    return 0


def stage_ssl(commit):
    arms = [
        ('ssl_book_nocl',
         overlay(book_champ(), runs='3', fusion='gate', use_cl='0', use_simgcl='0',
                 exp_tag='ssl_ablate_3run_book_nocl'),
         runs_done('book', 'product', 'ssl_ablate_3run_book_nocl', 3)),
        ('ssl_book_cl005',
         overlay(book_champ(), runs='3', fusion='gate', use_cl='1', cl_lambda='0.005',
                 cl_tau='0.2', use_simgcl='0', exp_tag='ssl_ablate_3run_book_cl005'),
         runs_done('book', 'product', 'ssl_ablate_3run_book_cl005', 3)),
        ('ssl_movie_nocl',
         overlay(movie_champ(), runs='3', fusion='gate', use_cl='0', use_simgcl='0',
                 early_stop='8', exp_tag='ssl_ablate_3run_movie_nocl'),
         runs_done('movie', 'product', 'ssl_ablate_3run_movie_nocl', 3)),
        ('ssl_movie_cl005',
         overlay(movie_champ(), runs='3', fusion='gate', use_cl='1', cl_lambda='0.005',
                 cl_tau='0.2', use_simgcl='0', early_stop='8',
                 exp_tag='ssl_ablate_3run_movie_cl005'),
         runs_done('movie', 'product', 'ssl_ablate_3run_movie_cl005', 3)),
        ('ssl_restaurant_nocl',
         overlay(restaurant_champ(), runs='3', fusion='gate', use_cl='0', use_simgcl='0',
                 exp_tag='ssl_ablate_3run_restaurant_nocl'),
         runs_done('restaurant', 'product', 'ssl_ablate_3run_restaurant_nocl', 3)),
        ('ssl_restaurant_cl256',
         overlay(restaurant_champ(), runs='3', fusion='gate', use_cl='1',
                 cl_lambda='0.02', cl_tau='0.2', cl_sample_size='256', use_simgcl='0',
                 exp_tag='ssl_ablate_3run_restaurant_cl256'),
         runs_done('restaurant', 'product', 'ssl_ablate_3run_restaurant_cl256', 3)),
    ]
    for name, pairs, skip in arms:
        rc = run_arm(name, _cmd(pairs), skip)
        if rc != 0:
            return rc
    note = summarize_ssl()
    if commit:
        git_save('feat: three-run SSL ablation on book movie restaurant',
                 [note, RESULT])
    print('STAGE_DONE 4 ssl-ablation', flush=True)
    return 0


def stage_paper(commit):
    kill = dict(
        att='product', fusion='fixed', use_cl='0', use_simgcl='0',
        lr_decay='1.0', early_stop='0', early_stop_min_epoch='0', runs='3',
        exp_tag='thesis_paper_product_sanity',
    )
    arms = [
        ('paper_book', overlay(book_champ(), **kill),
         runs_done('book', 'product', 'thesis_paper_product_sanity', 3)),
        ('paper_movie', overlay(movie_champ(), **kill),
         runs_done('movie', 'product', 'thesis_paper_product_sanity', 3)),
        ('paper_restaurant', overlay(restaurant_champ(), **kill),
         runs_done('restaurant', 'product', 'thesis_paper_product_sanity', 3)),
    ]
    for name, pairs, skip in arms:
        rc = run_arm(name, _cmd(pairs), skip)
        if rc != 0:
            return rc
    note = summarize_paper()
    if commit:
        git_save('feat: paper Product baseline replay on book movie restaurant',
                 [note, RESULT])
    print('STAGE_DONE 6 paper-baseline', flush=True)
    return 0


STAGES = (
    (1, 'interpret', stage_interpret),
    (2, 'product-screens', stage_screens),
    (4, 'ssl-ablation', stage_ssl),
    (6, 'paper-baseline', stage_paper),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='+', type=int, default=None,
                    help='subset of 1 2 4 6 (default: all four)')
    ap.add_argument('--commit', type=int, default=1,
                    help='git-commit result files + thesis notes after each stage')
    ap.add_argument('--summarize-only', type=int, default=0,
                    help='rewrite notes from existing result files; no training')
    args = ap.parse_args()
    wanted = args.only or [1, 2, 4, 6]
    unknown = [x for x in wanted if x not in (1, 2, 4, 6)]
    if unknown:
        sys.exit('unknown stage(s): %s  known: 1 2 4 6' % wanted)

    os.makedirs(NOTES, exist_ok=True)
    os.chdir(SRC)
    if args.summarize_only:
        mapping = {1: summarize_interpret, 2: summarize_screens,
                   4: summarize_ssl, 6: summarize_paper}
        for sid in wanted:
            mapping[sid]()
        print('SUMMARIZE_ONLY_DONE', flush=True)
        return

    lookup = dict((sid, fn) for sid, _name, fn in STAGES)
    for sid, name, _fn in STAGES:
        if sid not in wanted:
            continue
        print('STAGE_START %s %s' % (sid, name), flush=True)
        rc = lookup[sid](bool(args.commit))
        if rc != 0:
            sys.exit(rc)
    print('ALL_SHORT_TASKS_DONE', flush=True)


if __name__ == '__main__':
    main()
