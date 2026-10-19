import tensorflow as tf
import numpy as np
import scipy.io as sio
from model import KGCN
import os
import json
import time
import subprocess
import traceback


def _is_finite_number(x):
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _invalid_run_reason(show_ctr, show_topk, best_auc, best_f1, best_recall, k_list):
    # never treat a low-but-finite score as a failed run.
    if show_ctr:
        if not _is_finite_number(best_auc) or not _is_finite_number(best_f1):
            return 'non-finite auc/f1'
    if show_topk:
        if best_recall is None:
            return 'missing recall'
        arr = np.asarray(best_recall, dtype=float)
        if arr.ndim != 1 or arr.size != len(k_list) or not np.all(np.isfinite(arr)):
            return 'non-finite or incomplete recall'
    return None


def _nvidia_smi_used_bytes(gpu_id):
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits',
             '-i', str(int(gpu_id))],
            stderr=subprocess.STDOUT, timeout=8)
        line = out.decode('utf-8', errors='replace').strip().splitlines()[0].strip()
        mib = float(line)
        return int(mib * 1024 * 1024)
    except Exception:
        return None


def _tf_memory_bytes():
    try:
        info = tf.config.experimental.get_memory_info('GPU:0')
        peak = int(info.get('peak') or 0)
        current = int(info.get('current') or 0)
        return max(peak, current)
    except Exception:
        return None


def _tf_reset_peak():
    try:
        tf.config.experimental.reset_memory_stats('GPU:0')
    except Exception:
        return


def _sample_vram(gpu_id, samples):
    used = _nvidia_smi_used_bytes(gpu_id)
    if used is not None:
        samples.append(('nvidia-smi', used))
    tfb = _tf_memory_bytes()
    if tfb is not None:
        samples.append(('tf.get_memory_info', tfb))


def _peak_from_samples(samples):
    if not samples:
        return None, None, 'unavailable'
    peak = max(v for _, v in samples)
    sources = sorted(set(name for name, _ in samples))
    return int(peak), peak / (1024.0 * 1024.0), '+'.join(sources)


def _runs_json_path(args):
    return '../result/runs_{}_{}_{}.json'.format(args.model_type, args.dataset, _att_tag(args))


def _json_safe(x):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, np.ndarray):
        return [_json_safe(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def _dump_runs_json(args, payload):
    with open(_runs_json_path(args), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _payload_from_records(args, k_list, run_records, auc, f1_score, recall, precision, ndcg,
                          include_resources):
    payload = {
        'model_type': args.model_type, 'dataset': args.dataset, 'att': args.att,
        'exp_tag': getattr(args, 'exp_tag', ''), 'k_list': list(k_list),
        'runs': int(len(auc)),
        'auc': np.array(auc).tolist() if len(auc) else [],
        'f1': np.array(f1_score).tolist() if len(f1_score) else [],
        'recall': np.array(recall).tolist() if len(recall) else [],
        'precision': np.array(precision).tolist() if len(precision) else [],
    }
    if bool(getattr(args, 'report_ndcg', 0)) and len(ndcg) > 0 and ndcg[0] is not None:
        payload['ndcg'] = np.array(ndcg).tolist()
    if include_resources:
        payload['n_attempted'] = int(len(run_records))
        payload['n_valid'] = int(len(auc))
        payload['run_records'] = run_records
        # complete CLI snapshot for protocol runs (--log_resources 1 only)
        payload['config'] = {k: _json_safe(getattr(args, k)) for k in sorted(vars(args))}
    return payload


def train_n_runs(args, data, show_loss, show_topk, show_ctr):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]
    user_item_adj = data[9]

    auc = []
    f1_score = []
    precision = []
    recall = []
    ndcg = []  # Task 1: best test NDCG@K per run (at the best-eval-R@10 epoch)
    epochs_auc = []
    epochs_recall = []
    train_losses = []  # New: To store the loss values
    cl_losses = []  # Task 8: per-epoch average contrastive loss (parallel to train_losses)

    # Task 3: cold-start / data-sparsity eval state
    sparsity_runs = []
    sparsity_edges = sorted(eval(getattr(args, 'sparsity_buckets', '[5,10,20]')))
    user_pos_count = get_user_pos_count(train_data) if getattr(args, 'sparsity_eval', 0) else {}



    RUNS = args.runs

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    ckpt_save_path_1 = '../Pretrain/{}/{}/ctr/'.format(args.model_type, args.dataset)
    ckpt_save_path_2 = '../Pretrain/{}/{}/topk/'.format(args.model_type, args.dataset)

    if not os.path.exists(ckpt_save_path_1):
        os.makedirs(ckpt_save_path_1)
    if not os.path.exists(ckpt_save_path_2):
        os.makedirs(ckpt_save_path_2)

    if args.pretrain == 1:
        user_emb = np.load('../output/KGCN_{}_user_emb.npy'.format(args.dataset))
        entity_emb = np.load('../output/KGCN_{}_entity_emb.npy'.format(args.dataset))
        relation_emb = np.load('../output/KGCN_{}_relation_emb.npy'.format(args.dataset))
        pretrain = {'user': user_emb, 'entity': entity_emb, 'relation': relation_emb}
    else:
        pretrain = None

    args.seed += 1
    print('----------')
    print(args.seed)
    print('----------')

    log_res = bool(int(getattr(args, 'log_resources', 0) or 0))
    run_records = []
    k_list_test = eval(args.topk_k_list)

    for r in range(RUNS):
        seed_eval = int(args.seed)
        best_auc = 0
        best_f1 = 0
        best_recall = None
        best_precision = None
        best_ndcg = None
        best_auc_epoch = 0
        best_recall_epoch = 0
        train_seconds = 0.0
        vram_samples = []
        t_run0 = time.perf_counter()
        rec = {
            'run_index': r,
            'status': 'ok',
            'seed_eval': seed_eval,
            'seed_test': None,
            'error': None,
        }

        try:
            tf.compat.v1.reset_default_graph()

            model = KGCN(args, n_user, n_item, n_entity, n_relation, adj_entity, adj_relation, user_item_adj, pretrain=pretrain)

            saver_ckpt = tf.compat.v1.train.Saver()

            # Init to -1 so the first epoch always populates best_* (a degenerate run
            # where eval AUC/Recall@10 never exceeds 0 would otherwise leave best_recall
            # = None and crash the per-run np.array aggregation). No effect on normal runs.
            best_auc_eval = -1.0
            best_r10_eval = -1.0

            user_list_eval, train_record_eval, test_record_eval, \
                item_set_eval, k_list_eval = topk_settings(
                    show_topk, train_data, eval_data, n_item, seed=args.seed, args=args)
            args.seed += 1

            rec['seed_test'] = int(args.seed)
            user_list_test, train_record_test, test_record_test, \
                item_set_test, k_list_test = topk_settings(
                    show_topk, train_data, test_data, n_item, seed=args.seed, args=args)
            args.seed += 1

            idx_eval_r10 = recall_k_index(k_list_eval, 10)
            rank_ckpt = str(getattr(args, 'rank_ckpt', 'r10') or 'r10').strip().lower()
            if rank_ckpt not in ('r10', 'auc'):
                print('unknown --rank_ckpt %s; using r10' % rank_ckpt)
                rank_ckpt = 'r10'
            if rank_ckpt == 'auc':
                print('rank_ckpt: auc (test Recall@K + early-stop follow eval AUC)')

            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.compat.v1.Session(config=config) as sess:
                sess.run(tf.compat.v1.global_variables_initializer())

                if not (args.logging == 'save'):
                    ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_restore_path + 'checkpoint'))
                    if ckpt and ckpt.model_checkpoint_path:
                        saver_ckpt.restore(sess, ckpt.model_checkpoint_path)
                        print('================Done loading======================')
                else:
                    print('Initialized from scratch')

                if log_res:
                    _tf_reset_peak()
                    _sample_vram(args.gpu_id, vram_samples)

                for step in range(args.n_epochs):
                    np.random.shuffle(train_data)
                    start = 0

                    epoch_train_loss = []  # New: To store the loss for each epoch
                    epoch_cl_loss = []  # Task 8: contrastive loss per minibatch
                    epoch_simgcl_loss = []

                    lr_decay = float(getattr(args, 'lr_decay', 1.0))
                    cur_lr = args.lr * (lr_decay ** step) if lr_decay < 1.0 else None

                    t_tr0 = time.perf_counter()
                    while start + args.batch_size <= train_data.shape[0]:
                        _, loss, cl_loss, simgcl_loss = model.train(
                            sess, get_feed_dict(args, model, train_data, start, start + args.batch_size, lr=cur_lr))
                        epoch_train_loss.append(loss)
                        epoch_cl_loss.append(cl_loss)
                        epoch_simgcl_loss.append(simgcl_loss)
                        start += args.batch_size
                        if show_loss:
                            print(start, loss)
                    train_seconds += time.perf_counter() - t_tr0

                    avg_train_loss = np.mean(epoch_train_loss)  #  Calculate the average loss for the epoch
                    avg_cl_loss = float(np.mean(epoch_cl_loss)) if len(epoch_cl_loss) else 0.0
                    avg_simgcl_loss = float(np.mean(epoch_simgcl_loss)) if len(epoch_simgcl_loss) else 0.0
                    loss_bits = ['epoch %d    train_loss: %.4f' % (step, avg_train_loss)]
                    if getattr(args, 'use_cl', 0):
                        loss_bits.append('cl_loss: %.4f' % avg_cl_loss)
                    if getattr(args, 'use_simgcl', 0):
                        loss_bits.append('simgcl_loss: %.4f' % avg_simgcl_loss)
                    if cur_lr is not None:
                        loss_bits.append('lr: %.6f' % cur_lr)
                    print('    '.join(loss_bits))

                    train_losses.append(avg_train_loss)  # Store the average loss
                    cl_losses.append(avg_cl_loss)

                    with open('../result/{}_{}_train_loss_{}.txt'.format(args.model_type, args.dataset, _att_tag(args)), 'w') as f:
                        for index in range(len(train_losses)):
                            if getattr(args, 'use_cl', 0):
                                f.write('epoch:{:2d} train_loss:{:.4f} cl_loss:{:.4f}\n'.format(
                                    index, train_losses[index], cl_losses[index]))
                            else:
                                f.write('epoch:{:2d} train_loss:{:.4f}\n'.format(index, train_losses[index]))

                    # cache LightGCN/NGCF tables once for this epoch's CTR + top-K (eval only)
                    cf_cache = None
                    if getattr(args, 'fast_eval', 0) and model.supports_cf_cache():
                        cf_cache = model.cache_cf_embeddings(sess)

                    if show_ctr:
                        train_auc, train_f1 = ctr_eval(args, sess, model, train_data, args.batch_size, cf_cache)
                        eval_auc, eval_f1 = ctr_eval(args, sess, model, eval_data, args.batch_size, cf_cache)
                        test_auc, test_f1 = ctr_eval(args, sess, model, test_data, args.batch_size, cf_cache)

                        print('epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f'
                              % (step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))

                        eval_logging_ctr(args, step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1)
                        if eval_auc > best_auc_eval:
                            best_auc_epoch = step
                            best_auc_eval = eval_auc
                            best_auc = test_auc
                            best_f1 = test_f1

                            user_embedding, entity_embedding, relation_embedding = model.get_embeddings(sess)
                            sio.savemat('../output/{}_{}_emb.mat'.format(args.model_type, args.dataset), \
                                {'user': user_embedding, 'entity': entity_embedding, 'relation': relation_embedding})
                            np.save('../output/{}_{}_user_emb.npy'.format(args.model_type, args.dataset), user_embedding)
                            np.save('../output/{}_{}_entity_emb.npy'.format(args.model_type, args.dataset), entity_embedding)
                            np.save('../output/{}_{}_relation_emb.npy'.format(args.model_type, args.dataset), relation_embedding)
                            if getattr(args, 'pair_dump', 0) and cf_cache is not None:
                                try:
                                    dump_test_pairs(args, sess, model, test_data, args.batch_size, cf_cache, r, step)
                                except Exception as e:
                                    print('[pair_dump] skipped: %s' % e)

                    if show_topk:
                        eval_precision, eval_recall, eval_ndcg = topk_eval(
                            args, sess, model, user_list_eval, train_record_eval, test_record_eval,
                            item_set_eval, k_list_eval, args.batch_size, cf_cache)
                        test_precision, test_recall, test_ndcg = topk_eval(
                            args, sess, model, user_list_test, train_record_test, test_record_test,
                            item_set_test, k_list_test, args.batch_size, cf_cache)

                        eval_logging_topk(args, step, k_list_eval, eval_precision, eval_recall, test_precision, test_recall, eval_ndcg, test_ndcg)
                        # r10: snapshot ranking on eval R@10 (legacy, strict >).
                        # auc: same epoch as F1 so a run does not mix CTR@ep2 with R@10@ep6.
                        if rank_ckpt == 'auc':
                            if eval_recall[idx_eval_r10] > best_r10_eval:
                                best_r10_eval = eval_recall[idx_eval_r10]
                            if show_ctr and step == best_auc_epoch:
                                best_recall_epoch = step
                                best_recall = test_recall
                                best_precision = test_precision
                                best_ndcg = test_ndcg
                        elif eval_recall[idx_eval_r10] > best_r10_eval:
                            best_recall_epoch = step
                            best_r10_eval = eval_recall[idx_eval_r10]
                            best_recall = test_recall
                            best_precision = test_precision
                            best_ndcg = test_ndcg

                    # early stop: r10 + top-K -> eval R@10; otherwise eval AUC.
                    # warmup (--early_stop_min_epoch) avoids killing slow-start runs.
                    # if ranking never left zero, do not stop (book-style stuck-at-chance).
                    patience = int(getattr(args, 'early_stop', 0) or 0)
                    min_ep = int(getattr(args, 'early_stop_min_epoch', 0) or 0)
                    es_on_auc = (rank_ckpt == 'auc') or (not show_topk)
                    if patience > 0 and step >= min_ep:
                        if (not es_on_auc) and show_topk and (step - best_recall_epoch) >= patience:
                            if best_r10_eval <= 0.0:
                                if step == min_ep or step % 5 == 0:
                                    print('Early stop deferred: eval R@10 still %.4f at epoch %d '
                                          '(need ranking signal before patience applies)'
                                          % (best_r10_eval, step))
                            else:
                                print('Early stopping at epoch %d: no eval R@10 improvement for %d epochs '
                                      '(best epoch %d, eval R@10 %.4f)'
                                      % (step, patience, best_recall_epoch, best_r10_eval))
                                break
                        if es_on_auc and show_ctr and (step - best_auc_epoch) >= patience:
                            print('Early stopping at epoch %d: no eval AUC improvement for %d epochs '
                                  '(best epoch %d, eval auc %.4f)'
                                  % (step, patience, best_auc_epoch, best_auc_eval))
                            break

                    if log_res:
                        _sample_vram(args.gpu_id, vram_samples)

                # Task 3: per-bucket cold-start eval using the final-epoch model
                if getattr(args, 'sparsity_eval', 0) and show_topk:
                    end_cache = None
                    if getattr(args, 'fast_eval', 0) and model.supports_cf_cache():
                        end_cache = model.cache_cf_embeddings(sess)
                    sparsity_runs.append(topk_eval_buckets(
                        args, sess, model, user_list_test, train_record_test, test_record_test,
                        item_set_test, k_list_test, args.batch_size, user_pos_count, sparsity_edges, end_cache))

                # Task 6: attention interpretability case study (first run only)
                if getattr(args, 'interpret', 0) and show_topk and r == 0:
                    export_attention_case_study(
                        args, sess, model, train_record_test, test_record_test, n_item)

                if log_res:
                    _sample_vram(args.gpu_id, vram_samples)

            invalid = _invalid_run_reason(
                show_ctr, show_topk, best_auc, best_f1, best_recall, k_list_test)
            if log_res and invalid:
                rec['status'] = 'invalid'
                rec['error'] = invalid
                print('[log_resources] run %d invalid: %s' % (r, invalid))
            else:
                rec['status'] = 'ok'
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if not log_res:
                raise
            rec['status'] = 'crashed'
            rec['error'] = '{}: {}'.format(type(e).__name__, str(e)[:500])
            traceback.print_exc()
            print('[log_resources] run %d crashed: %s' % (r, rec['error']))

        rec['train_seconds'] = float(train_seconds)
        rec['total_seconds'] = float(time.perf_counter() - t_run0)
        rec['eval_seconds'] = float(rec['total_seconds'] - rec['train_seconds'])
        if log_res:
            peak_b, peak_mb, src = _peak_from_samples(vram_samples)
        else:
            peak_b, peak_mb, src = None, None, 'not_requested'
        rec['peak_vram_bytes'] = peak_b
        rec['peak_vram_mb'] = None if peak_mb is None else round(float(peak_mb), 1)
        rec['vram_source'] = src
        rec['best_auc_epoch'] = int(best_auc_epoch)
        rec['best_recall_epoch'] = int(best_recall_epoch)
        rec['rank_ckpt'] = str(getattr(args, 'rank_ckpt', 'r10') or 'r10')
        rec['auc'] = _json_safe(best_auc) if rec['status'] == 'ok' else None
        rec['f1'] = _json_safe(best_f1) if rec['status'] == 'ok' else None
        rec['recall'] = _json_safe(best_recall) if rec['status'] == 'ok' else None
        rec['precision'] = _json_safe(best_precision) if rec['status'] == 'ok' else None
        rec['ndcg'] = _json_safe(best_ndcg) if rec['status'] == 'ok' else None

        if rec['status'] == 'ok':
            if show_ctr:
                auc.append(best_auc)
                f1_score.append(best_f1)
                recall.append(best_recall)
                precision.append(best_precision)
                ndcg.append(best_ndcg)
            if show_topk:
                epochs_recall.append(best_recall_epoch)
                epochs_auc.append(best_auc_epoch)
        elif log_res:
            print('[log_resources] excluding run %d from mean (%s)' % (r, rec['status']))

        run_records.append(rec)
        if log_res:
            _dump_runs_json(
                args, _payload_from_records(
                    args, k_list_test, run_records, auc, f1_score, recall, precision, ndcg, True))

    if show_ctr and show_topk:
        n_valid = len(auc)
        if log_res:
            print('[log_resources] valid runs: {}/{}'.format(n_valid, RUNS))
        if n_valid == 0:
            print('[log_resources] no valid runs to aggregate; skipping mean±std')
            if log_res:
                _dump_runs_json(
                    args, _payload_from_records(
                        args, k_list_test, run_records, auc, f1_score, recall, precision, ndcg, True))
        else:
            auc = np.array(auc)
            f1_score = np.array(f1_score)
            recall = np.array(recall)
            precision = np.array(precision)

            print(auc)
            print(f1_score)
            print(recall)
            print(precision)

            printings_auc = 'average auc: {:.4f}+\\-{:.4f}'.format(np.mean(auc), np.std(auc))
            printings_f1 = 'average f1 : {:.4f}+\\-{:.4f}'.format(np.mean(f1_score), np.std(f1_score))

            printings_recall = 'average recall   : '
            for i in range(len(k_list_test)):
                printings_recall = printings_recall + '{:.4f}+\\-{:.4f} '.format(np.mean(recall[:, i]), np.std(recall[:, i]))
            printings_precision = 'average precision: '
            for i in range(len(k_list_test)):
                printings_precision = printings_precision + '{:.4f}+\\-{:.4f} '.format(np.mean(precision[:, i]), np.std(precision[:, i]))

            print(printings_auc)
            print(printings_f1)
            print(printings_recall)
            print(printings_precision)

            printings_ndcg = ''
            if bool(getattr(args, 'report_ndcg', 0)) and len(ndcg) > 0 and ndcg[0] is not None:
                ndcg_arr = np.array(ndcg)
                printings_ndcg = 'average ndcg     : '
                for i in range(len(k_list_test)):
                    printings_ndcg = printings_ndcg + '{:.4f}+\\-{:.4f} '.format(np.mean(ndcg_arr[:, i]), np.std(ndcg_arr[:, i]))
                print(printings_ndcg)

            printings_paper = format_paper_recall_block(k_list_test, recall, precision, ndcg if printings_ndcg else None)
            print(printings_paper)
            logging_final(args, printings_auc, printings_f1, printings_recall, printings_precision, printings_paper, printings_ndcg)

            if log_res:
                ok_recs = [x for x in run_records if x.get('status') == 'ok']
                tr = [x['train_seconds'] for x in ok_recs if x.get('train_seconds') is not None]
                tot = [x['total_seconds'] for x in ok_recs if x.get('total_seconds') is not None]
                vrm = [x['peak_vram_mb'] for x in ok_recs if x.get('peak_vram_mb') is not None]
                if tr:
                    print('average train_seconds: {:.1f}+\\-{:.1f}'.format(np.mean(tr), np.std(tr)))
                if tot:
                    print('average total_seconds: {:.1f}+\\-{:.1f}'.format(np.mean(tot), np.std(tot)))
                if vrm:
                    print('average peak_vram_mb: {:.1f}+\\-{:.1f}'.format(np.mean(vrm), np.std(vrm)))
                else:
                    print('average peak_vram_mb: unavailable')

            # Task 7: dump per-run results (machine-readable) for the paired significance test.
            # Additive (a new file); existing logs are unchanged.
            runs_payload = _payload_from_records(
                args, k_list_test, run_records, auc, f1_score, recall, precision, ndcg, log_res)
            _dump_runs_json(args, runs_payload)

    if getattr(args, 'sparsity_eval', 0):
        logging_sparsity(args, sparsity_runs, k_list_test, sparsity_edges)
    print("average train_Loss", np.mean(train_losses))  # New: Print average losses after training

'''
***********************************************
logging functions
'''
def _att_tag(args):
    # Task 2: append an optional experiment tag so ablation variants that share a
    # model_type write to separate result files. Empty tag => legacy filenames.
    tag = getattr(args, 'exp_tag', '') or ''
    return args.att + ('_' + tag if tag else '')


def logging_final(args, auc, f1_score, recall, precision, paper_block='', ndcg=''):
    fid = open('../result/{}_{}_{}.txt'.format(args.model_type, args.dataset, _att_tag(args)), 'a')

    fid.write('\n******************************************************************************\n')
    params = eval(args.params)
    settings = ''
    for i in range(len(params)):
        settings = settings + params[i] + ':{} '.format(eval('args.'+params[i]))
    fid.write(settings+'\n')
    fid.write('******************************************************************************\n')

    fid.write(auc+'\n')
    fid.write(f1_score+'\n')
    fid.write(recall+'\n')
    fid.write(precision+'\n')
    if ndcg:
        fid.write(ndcg+'\n')
    if paper_block:
        fid.write(paper_block + '\n')

    fid.close()


def eval_logging_topk(args, epoch, k_list, eval_pr, eval_re, test_pr, test_re, eval_ndcg=None, test_ndcg=None):
    report_ndcg = bool(getattr(args, 'report_ndcg', 0)) and eval_ndcg is not None and test_ndcg is not None
    fid = open('../result/{}_{}_topk_{}.txt'.format(args.model_type, args.dataset, _att_tag(args)), 'a')

    if epoch == 0:
        fid.write('\n******************************************************************************\n')
        params = eval(args.params)
        settings = ''
        for i in range(len(params)):
            settings = settings + params[i] + ':{} '.format(eval('args.'+params[i]))
        fid.write(settings+'\n')
        fid.write('k_list (Recall@K): {}\n'.format(k_list))
        fid.write('******************************************************************************\n')

    def _fmt(vals):
        s = ''
        for j in vals:
            s = s + '{:.4f}\t'.format(j)
        return s

    # eval line: precision, recall (+ ndcg). Layout matches the legacy format; the
    # ndcg block is appended only when --report_ndcg is set, so legacy runs are
    # byte-for-byte unchanged.
    eval_line = 'epoch:{:2d} eval precision: '.format(epoch) + _fmt(eval_pr) + 'eval recall: ' + _fmt(eval_re)
    if report_ndcg:
        eval_line = eval_line + 'eval ndcg: ' + _fmt(eval_ndcg)
    fid.write(eval_line + '\n')

    test_line = 'epoch:{:2d} test precision: '.format(epoch) + _fmt(test_pr) + 'test recall: ' + _fmt(test_re)
    if report_ndcg:
        test_line = test_line + 'test ndcg: ' + _fmt(test_ndcg)
    fid.write(test_line + '\n')

    fid.write('epoch:{:2d} test {}'.format(epoch, format_paper_recall_line(k_list, test_re)))
    if report_ndcg:
        fid.write(format_paper_ndcg_line(k_list, test_ndcg))
    fid.write('\n')
    fid.close()

def eval_logging_ctr(args, epoch, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1):
    fid = open('../result/{}_{}_ctr_{}.txt'.format(args.model_type, args.dataset, _att_tag(args)), 'a')

    if epoch == 0:
        fid.write('\n******************************************************************************\n')
        params = eval(args.params)
        settings = ''
        for i in range(len(params)):
            settings = settings + params[i] + ':{} '.format(eval('args.'+params[i]))
        fid.write(settings+'\n')
        fid.write('******************************************************************************\n')

    fid.write('epoch:{:2d} train_auc:{:.4f} train_f1:{:.4f} '.format(epoch, train_auc, train_f1) +
              'eval_auc:{:.4f} eval_f1:{:.4f} '.format(eval_auc, eval_f1) +
              'test_auc:{:.4f} test_f1:{:.4f}\n'.format(test_auc, test_f1))

    fid.close()


def logging_sparsity(args, sparsity_runs, k_list, edges):
    # Task 3: aggregate per-bucket metrics (mean+-std over runs) and append to
    # result/{model_type}_{dataset}_sparsity_{att}.txt
    if not sparsity_runs:
        return
    names = bucket_names(edges)
    fid = open('../result/{}_{}_sparsity_{}.txt'.format(args.model_type, args.dataset, _att_tag(args)), 'a')

    fid.write('\n******************************************************************************\n')
    params = eval(args.params)
    settings = ''
    for i in range(len(params)):
        settings = settings + params[i] + ':{} '.format(eval('args.'+params[i]))
    fid.write(settings+'\n')
    fid.write('sparsity_buckets (pos train interactions): {}   k_list: {}   runs: {}\n'.format(
        edges, k_list, len(sparsity_runs)))
    fid.write('******************************************************************************\n')

    tagmap = {'recall': 'R', 'ndcg': 'N', 'precision': 'P'}
    for b in names:
        avg_count = float(np.mean([r[b]['count'] for r in sparsity_runs]))
        line = 'bucket:{:<7} users:{:.0f}  '.format(b, avg_count)
        for metric in ['recall', 'ndcg', 'precision']:
            for j, k in enumerate(k_list):
                vals = [r[b][metric][j] for r in sparsity_runs]
                line += '{}@{}:{:.4f}+-{:.4f}  '.format(tagmap[metric], k, np.nanmean(vals), np.nanstd(vals))
        fid.write(line + '\n')
    fid.close()
    print('sparsity per-bucket results written to ../result/{}_{}_sparsity_{}.txt'.format(
        args.model_type, args.dataset, _att_tag(args)))


def export_attention_case_study(args, sess, model, train_record, test_record, n_item):
    # Task 6: pick a user, rank items for them, and export the personalized knowledge-aware
    # attention over the KG neighbors of the top recommended items (the "why" of each rec).
    if getattr(model, 'kg_attention', None) is None:
        print('[interpret] no KG attention available (need a KGCN_* model with a personalized --att); skipped')
        return

    candidates = set(train_record.keys()) & set(test_record.keys())
    if not candidates:
        print('[interpret] no train/test user overlap; skipped')
        return
    user = getattr(args, 'interpret_user', -1)
    if user is None or user < 0 or user not in candidates:
        user = max(candidates, key=lambda u: len(test_record[u]))  # most active held-out user

    item_set = set(range(n_item))
    item_sorted = _score_user_items(args, sess, model, user, train_record, item_set, args.batch_size)
    topk = max(1, int(getattr(args, 'interpret_topk', 5)))
    top_items = item_sorted[:topk]

    # one padded batch so the static-graph reshapes (which use batch_size) are satisfied
    bs = args.batch_size
    padded = (list(top_items) + [top_items[-1]] * (bs - len(top_items)))[:bs]
    feed = {model.user_indices: [user] * bs,
            model.item_indices: padded,
            model.node_dropout: [0.] * len(eval(args.layer_size)),
            model.mess_dropout: [0.] * len(eval(args.layer_size))}
    _, scores = model.get_scores(sess, feed)
    _, att = model.get_attention(sess, feed)            # [bs, 1, n_neighbor]
    att = np.reshape(np.array(att), (bs, -1))           # [bs, n_neighbor]
    score_map = {int(it): float(sc) for it, sc in zip(padded, scores)}

    adj_entity, adj_relation = model.adj_entity, model.adj_relation
    recs = []
    for rank, item in enumerate(top_items):
        neighbors = [{'entity': int(adj_entity[item][j]),
                      'relation': int(adj_relation[item][j]),
                      'attention': float(att[rank][j])}
                     for j in range(adj_entity.shape[1])]
        neighbors.sort(key=lambda d: d['attention'], reverse=True)
        recs.append({'rank': rank + 1, 'item': int(item),
                     'score': score_map.get(int(item), float('nan')),
                     'in_test': bool(item in test_record[user]),
                     'kg_neighbors': neighbors})

    payload = {'dataset': args.dataset, 'model_type': args.model_type, 'att': args.att,
               'user': int(user), 'n_neighbor': int(args.neighbor_sample_size),
               'user_train_items': sorted(int(x) for x in train_record[user]),
               'user_test_items': sorted(int(x) for x in test_record[user]),
               'recommendations': recs}

    base = '../result/{}_{}_attention_{}'.format(args.model_type, args.dataset, _att_tag(args))
    with open(base + '.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    with open(base + '.txt', 'w', encoding='utf-8') as f:
        f.write('Attention case study -- dataset:{} model:{} att:{}\n'.format(
            args.dataset, args.model_type, args.att))
        f.write('user:{}  (train items:{}  test items:{})\n'.format(
            user, len(train_record[user]), len(test_record[user])))
        f.write('top-{} recommended items and their KG-neighbor attention:\n\n'.format(len(top_items)))
        for rec in recs:
            f.write('rank {}: item {}  score={:.4f}  {}\n'.format(
                rec['rank'], rec['item'], rec['score'], '[in test]' if rec['in_test'] else ''))
            for nb in rec['kg_neighbors']:
                f.write('    neighbor entity {:>7}  relation {:>3}  attention {:.4f}\n'.format(
                    nb['entity'], nb['relation'], nb['attention']))
            f.write('\n')
    print('attention case study (user {}) written to {}.json / .txt'.format(user, base))
'''
***********************************************
'''

def recall_k_index(k_list, k):
    if k in k_list:
        return k_list.index(k)
    return min(3, len(k_list) - 1) if len(k_list) > 0 else 0


def format_paper_recall_line(k_list, recall_vec):
    line = ''
    for pk in (2, 10, 50, 100):
        if pk in k_list:
            i = k_list.index(pk)
            line += 'R@{}:{:.4f}\t'.format(pk, recall_vec[i])
    return line.rstrip() if line else '(add K to topk_k_list for paper R@2/10/50/100)\t'


def format_paper_ndcg_line(k_list, ndcg_vec):
    line = ''
    for pk in (2, 10, 50, 100):
        if pk in k_list:
            i = k_list.index(pk)
            line += 'N@{}:{:.4f}\t'.format(pk, ndcg_vec[i])
    return ('\t' + line.rstrip()) if line else ''


def format_paper_recall_block(k_list, recall_runs, precision_runs, ndcg_runs=None):
    ndcg_arr = np.array(ndcg_runs) if ndcg_runs is not None else None
    lines = ['paper_style (mean+-std over runs, test set):']
    for pk in (2, 10, 50, 100):
        if pk not in k_list:
            continue
        i = k_list.index(pk)
        rm = np.mean(recall_runs[:, i])
        rs = np.std(recall_runs[:, i])
        pm = np.mean(precision_runs[:, i])
        ps = np.std(precision_runs[:, i])
        line = '  R@{} recall: {:.4f}+-{:.4f}  precision: {:.4f}+-{:.4f}'.format(pk, rm, rs, pm, ps)
        if ndcg_arr is not None:
            line += '  ndcg: {:.4f}+-{:.4f}'.format(np.mean(ndcg_arr[:, i]), np.std(ndcg_arr[:, i]))
        lines.append(line)
    return '\n'.join(lines)


def topk_settings(show_topk, train_data, test_data, n_item, seed=555, args=None):
    np.random.seed(seed)
    if show_topk:
        user_num = getattr(args, 'topk_user_num', 100) if args is not None else 100
        k_list = list(eval(getattr(args, 'topk_k_list', '[1,2,5,10,20,50,100]'))) if args is not None else [1, 2, 5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if user_num is None or user_num <= 0:
            user_list = np.array(user_list, dtype=object)
        elif len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        else:
            user_list = np.array(user_list, dtype=object)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(args, model, data, start, end, flag=True, lr=None):
    if flag:
        feed_dict = {model.user_indices: data[start:end, 0],
                     model.item_indices: data[start:end, 1],
                     model.labels: data[start:end, 2],
                     model.node_dropout: eval(args.node_dropout),
                     model.mess_dropout: eval(args.mess_dropout)}
    else:
        feed_dict = {model.user_indices: data[start:end, 0],
                     model.item_indices: data[start:end, 1],
                     model.labels: data[start:end, 2],
                     model.node_dropout: [0.]*len(eval(args.layer_size)),
                     model.mess_dropout: [0.]*len(eval(args.layer_size))}
    if lr is not None:
        feed_dict[model.lr_ph] = lr
    return feed_dict


def dump_test_pairs(args, sess, model, data, batch_size, cache, run_idx, epoch):
    # Reviewer 4 analysis (--pair_dump 1): per-pair test label, score and fusion weight g
    # at the current best eval-AUC epoch, scored exactly as eval_cached does.
    users, items, labels, scores, gates = [], [], [], [], []
    start = 0
    while start + batch_size <= data.shape[0]:
        feed = get_feed_dict(args, model, data, start, start + batch_size, flag=False)
        u = np.asarray(feed[model.user_indices], dtype=np.int64)
        i = np.asarray(feed[model.item_indices], dtype=np.int64)
        item_cf = cache['ia'][i]
        item_kg = sess.run(model.item_embeddings_kg, feed)
        if cache.get('gate_W') is not None:
            logit = np.concatenate([item_cf, item_kg], axis=-1).dot(cache['gate_W']) + cache['gate_b']
            g = model._sigmoid_np(logit).reshape(-1)
        else:
            g = np.full(len(i), float(model.alpha), dtype=np.float32)
        item_final = model._fuse_item_embeddings_np(item_cf, item_kg, cache)
        s = model._sigmoid_np(np.sum(cache['ua'][u] * item_final, axis=1))
        users.append(u)
        items.append(i)
        labels.append(np.asarray(feed[model.labels]))
        scores.append(s)
        gates.append(g)
        start += batch_size
    path = '../result/pairs_{}_{}_run{}.npz'.format(args.dataset, args.exp_tag, run_idx)
    np.savez_compressed(path, user=np.concatenate(users), item=np.concatenate(items),
                        label=np.concatenate(labels), score=np.concatenate(scores),
                        gate=np.concatenate(gates), epoch=np.int64(epoch))


def ctr_eval(args, sess, model, data, batch_size, cf_cache=None):
    start = 0
    auc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        feed = get_feed_dict(args, model, data, start, start + batch_size, flag=False)
        if cf_cache is not None:
            auc, f1 = model.eval_cached(sess, feed, cf_cache)
        else:
            auc, f1 = model.eval(sess, feed)
        auc_list.append(auc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list))


def ndcg_at_k(item_sorted, test_set, k):
    # NDCG@k with binary relevance (paper Section 3-4): standard DCG/IDCG with a
    # log2 rank discount. rank 0 -> log2(2)=1, so the top item is undiscounted.
    dcg = 0.0
    for rank, item in enumerate(item_sorted[:k]):
        if item in test_set:
            dcg += 1.0 / np.log2(rank + 2)
    n_rel = min(len(test_set), k)
    idcg = 0.0
    for rank in range(n_rel):
        idcg += 1.0 / np.log2(rank + 2)
    return dcg / idcg if idcg > 0 else 0.0


def _score_user_items(args, sess, model, user, train_record, item_set, batch_size, cf_cache=None):
    # Rank all candidate items (catalog minus the user's training items) by model score.
    # Shared by the standard top-K eval and the cold-start per-bucket eval (Task 3).
    test_item_list = list(item_set - train_record[user])
    item_score_map = dict()
    drop0 = [0.] * len(eval(args.layer_size))
    start = 0
    while start + batch_size <= len(test_item_list):
        feed = {model.user_indices: [user] * batch_size,
                model.item_indices: test_item_list[start:start + batch_size],
                model.node_dropout: drop0,
                model.mess_dropout: drop0}
        if cf_cache is not None:
            items, scores = model.get_scores_cached(sess, feed, cf_cache)
        else:
            items, scores = model.get_scores(sess, feed)
        for item, score in zip(items, scores):
            item_score_map[item] = score
        start += batch_size

    # padding the last incomplete minibatch if exists
    if start < len(test_item_list):
        feed = {model.user_indices: [user] * batch_size,
                model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                        batch_size - len(test_item_list) + start),
                model.node_dropout: drop0,
                model.mess_dropout: drop0}
        if cf_cache is not None:
            items, scores = model.get_scores_cached(sess, feed, cf_cache)
        else:
            items, scores = model.get_scores(sess, feed)
        for item, score in zip(items, scores):
            item_score_map[item] = score

    item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
    return [i[0] for i in item_score_pair_sorted]


def topk_eval(args, sess, model, user_list, train_record, test_record, item_set, k_list, batch_size, cf_cache=None):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    for user in user_list:
        item_sorted = _score_user_items(
            args, sess, model, user, train_record, item_set, batch_size, cf_cache)
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))
            ndcg_list[k].append(ndcg_at_k(item_sorted, test_record[user], k))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]

    return precision, recall, ndcg


'''
***********************************************
Task 3: cold-start / data-sparsity per-bucket evaluation
'''


def bucket_names(edges):
    names = ['<{}'.format(edges[0])]
    for i in range(len(edges) - 1):
        names.append('{}-{}'.format(edges[i], edges[i + 1]))
    names.append('{}+'.format(edges[-1]))
    return names


def bucket_label(count, edges):
    if count < edges[0]:
        return '<{}'.format(edges[0])
    for i in range(len(edges) - 1):
        if edges[i] <= count < edges[i + 1]:
            return '{}-{}'.format(edges[i], edges[i + 1])
    return '{}+'.format(edges[-1])


def get_user_pos_count(train_data):
    # profile size = number of positive (label==1) training interactions per user
    counts = dict()
    for interaction in train_data:
        if interaction[2] == 1:
            counts[interaction[0]] = counts.get(interaction[0], 0) + 1
    return counts


def topk_eval_buckets(args, sess, model, user_list, train_record, test_record, item_set,
                      k_list, batch_size, user_pos_count, edges, cf_cache=None):
    names = bucket_names(edges)
    prec = {b: {k: [] for k in k_list} for b in names}
    rec = {b: {k: [] for k in k_list} for b in names}
    ndc = {b: {k: [] for k in k_list} for b in names}
    counts = {b: 0 for b in names}

    for user in user_list:
        b = bucket_label(user_pos_count.get(user, 0), edges)
        item_sorted = _score_user_items(
            args, sess, model, user, train_record, item_set, batch_size, cf_cache)
        counts[b] += 1
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            prec[b][k].append(hit_num / k)
            rec[b][k].append(hit_num / len(test_record[user]))
            ndc[b][k].append(ndcg_at_k(item_sorted, test_record[user], k))

    out = {}
    for b in names:
        out[b] = {
            'count': counts[b],
            'precision': [float(np.mean(prec[b][k])) if prec[b][k] else float('nan') for k in k_list],
            'recall': [float(np.mean(rec[b][k])) if rec[b][k] else float('nan') for k in k_list],
            'ndcg': [float(np.mean(ndc[b][k])) if ndc[b][k] else float('nan') for k in k_list],
        }
    return out


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
