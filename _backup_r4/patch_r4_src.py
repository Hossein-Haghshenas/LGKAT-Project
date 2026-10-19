import io
def rd(p):
    return io.open(p, encoding='utf-8', newline='').read()
def wr(p, s):
    io.open(p, 'w', encoding='utf-8', newline='').write(s)

# main.py: new default-off flag
p = 'src/main.py'
s = rd(p)
anchor = "parser.add_argument('--interpret_topk', type=int, default=5,\r\n                    help='number of top recommended items to explain')\r\n"
assert s.count(anchor) == 1
add = ("\r\n# Reviewer 4 analysis: at every new best eval-AUC epoch, save each test pair's user, item,\r\n"
       "# label, predicted score and fusion weight g (alpha for fixed fusion) to\r\n"
       "# result/pairs_{dataset}_{exp_tag}_run{r}.npz. Needs --fast_eval 1. Off by default.\r\n"
       "parser.add_argument('--pair_dump', type=int, default=0,\r\n"
       "                    help='1: dump per-pair test scores and gate values at the best eval-AUC epoch')\r\n")
assert 'pair_dump' not in s
s = s.replace(anchor, anchor + add)
wr(p, s)

# train_n_runs.py: dump helper + call at the best eval-AUC epoch
p = 'src/train_n_runs.py'
s = rd(p)
assert 'pair_dump' not in s
call_anchor = ("                            np.save('../output/{}_{}_relation_emb.npy'.format(args.model_type, args.dataset), relation_embedding)\r\n")
assert s.count(call_anchor) == 1
call = ("                            if getattr(args, 'pair_dump', 0) and cf_cache is not None:\r\n"
        "                                dump_test_pairs(args, sess, model, test_data, args.batch_size, cf_cache, r, step)\r\n")
s = s.replace(call_anchor, call_anchor + call)
fn_anchor = "def ctr_eval(args, sess, model, data, batch_size, cf_cache=None):\r\n"
assert s.count(fn_anchor) == 1
fn = '''def dump_test_pairs(args, sess, model, data, batch_size, cache, run_idx, epoch):
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


'''.replace('\n', '\r\n')
s = s.replace(fn_anchor, fn + fn_anchor)
wr(p, s)
print('ok')
