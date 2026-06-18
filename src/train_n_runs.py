import tensorflow as tf
import numpy as np
import scipy.io as sio
from model import KGCN
from evaluation import ranking_metrics_at_ks, write_structured_results
import os


def train_n_runs(args, data, show_loss, show_topk, show_ctr):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]
    user_item_adj = data[9]

    auc = []
    f1_score = []
    precision = []
    recall = []
    ndcg = []  # Phase 1: per-run best test NDCG@K (parallel to recall/precision)
    epochs_auc = []
    epochs_recall = []
    train_losses = []  # New: To store the loss values



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

    for r in range(RUNS):
        tf.compat.v1.reset_default_graph()

        model = KGCN(args, n_user, n_item, n_entity, n_relation, adj_entity, adj_relation, user_item_adj, pretrain=pretrain)

        saver_ckpt = tf.compat.v1.train.Saver()

        best_auc = 0
        best_f1 = 0
        best_recall = None
        best_precision = None
        best_ndcg = None

        best_auc_epoch = 0
        best_recall_epoch = 0

        best_auc_eval = 0
        best_r10_eval = 0

        user_list_eval, train_record_eval, test_record_eval, \
            item_set_eval, k_list_eval = topk_settings(
                show_topk, train_data, eval_data, n_item, seed=args.seed, args=args)
        args.seed += 1

        user_list_test, train_record_test, test_record_test, \
            item_set_test, k_list_test = topk_settings(
                show_topk, train_data, test_data, n_item, seed=args.seed, args=args)
        args.seed += 1

        idx_eval_r10 = recall_k_index(k_list_eval, 10)

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

            for step in range(args.n_epochs):
                np.random.shuffle(train_data)
                start = 0

                epoch_train_loss = []  # New: To store the loss for each epoch

                while start + args.batch_size <= train_data.shape[0]:
                    _, loss = model.train(sess, get_feed_dict(args, model, train_data, start, start + args.batch_size))
                    epoch_train_loss.append(loss)
                    start += args.batch_size
                    if show_loss:
                        print(start, loss)

                avg_train_loss = np.mean(epoch_train_loss)  #  Calculate the average loss for the epoch
                print('epoch %d    train_loss: %.4f' % (step, avg_train_loss))  # New: Print the average loss

                train_losses.append(avg_train_loss)  # Store the average loss

                with open('../result/{}_{}_train_loss_{}.txt'.format(args.model_type, args.dataset, args.att), 'w') as f:
                    for index, avg_train_loss in enumerate(train_losses):
                        f.write('epoch:{:2d} train_loss:{:.4f}\n'.format(index, avg_train_loss))

                if show_ctr:
                    train_auc, train_f1 = ctr_eval(args, sess, model, train_data, args.batch_size)
                    eval_auc, eval_f1 = ctr_eval(args, sess, model, eval_data, args.batch_size)
                    test_auc, test_f1 = ctr_eval(args, sess, model, test_data, args.batch_size)

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

                if show_topk:
                    eval_precision, eval_recall, eval_ndcg = topk_eval(
                        args, sess, model, user_list_eval, train_record_eval, test_record_eval, item_set_eval, k_list_eval, args.batch_size)
                    test_precision, test_recall, test_ndcg = topk_eval(
                        args, sess, model, user_list_test, train_record_test, test_record_test, item_set_test, k_list_test, args.batch_size)

                    eval_logging_topk(args, step, k_list_eval, eval_precision, eval_recall, test_precision, test_recall, eval_ndcg, test_ndcg)
                    if eval_recall[idx_eval_r10] > best_r10_eval:
                        best_recall_epoch = step
                        best_r10_eval = eval_recall[idx_eval_r10]
                        best_recall = test_recall
                        best_precision = test_precision
                        best_ndcg = test_ndcg

        if show_ctr:
            auc.append(best_auc)
            f1_score.append(best_f1)
            recall.append(best_recall)
            precision.append(best_precision)
            ndcg.append(best_ndcg)

        if show_topk:
            epochs_recall.append(best_recall_epoch)
            epochs_auc.append(best_auc_epoch)

    if show_ctr and show_topk:
        auc = np.array(auc)
        f1_score = np.array(f1_score)
        recall = np.array(recall)
        precision = np.array(precision)

        report_ndcg = bool(getattr(args, 'report_ndcg', 0)) and len(ndcg) > 0 and ndcg[0] is not None
        ndcg_arr = np.array(ndcg) if report_ndcg else None

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

        printings_ndcg = ''
        if ndcg_arr is not None:
            printings_ndcg = 'average ndcg     : '
            for i in range(len(k_list_test)):
                printings_ndcg = printings_ndcg + '{:.4f}+\\-{:.4f} '.format(np.mean(ndcg_arr[:, i]), np.std(ndcg_arr[:, i]))

        print(printings_auc)
        print(printings_f1)
        print(printings_recall)
        print(printings_precision)
        if printings_ndcg:
            print(printings_ndcg)

        printings_paper = format_paper_recall_block(k_list_test, recall, precision, ndcg_arr)
        print(printings_paper)
        logging_final(args, printings_auc, printings_f1, printings_recall, printings_precision, printings_paper, printings_ndcg)

        if bool(getattr(args, 'structured_out', 0)):
            try:
                jp, cp = write_structured_results(
                    args, k_list_test, auc, f1_score, precision, recall, ndcg_runs=ndcg_arr)
                print('structured results written:\n  {}\n  {}'.format(jp, cp))
            except Exception as e:
                print('[structured_out] skipped (%s)' % e)
    print("average train_Loss", np.mean(train_losses))  # New: Print average losses after training

'''
***********************************************
logging functions
'''
def logging_final(args, auc, f1_score, recall, precision, paper_block='', ndcg=''):
    fid = open('../result/{}_{}_{}.txt'.format(args.model_type, args.dataset, args.att), 'a')

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


def eval_logging_topk(args, epoch, k_list, eval_pr, eval_re, test_pr, test_re, eval_nd=None, test_nd=None):
    report_ndcg = bool(getattr(args, 'report_ndcg', 0)) and eval_nd is not None and test_nd is not None
    fid = open('../result/{}_{}_topk_{}.txt'.format(args.model_type, args.dataset, args.att), 'a')

    if epoch == 0:
        fid.write('\n******************************************************************************\n')
        params = eval(args.params)
        settings = ''
        for i in range(len(params)):
            settings = settings + params[i] + ':{} '.format(eval('args.'+params[i]))
        fid.write(settings+'\n')
        fid.write('k_list (Recall@K): {}\n'.format(k_list))
        fid.write('******************************************************************************\n')

    result_type = ['epoch:{:2d} eval precision: '.format(epoch), 'eval recall: '.format(epoch),
                   'epoch:{:2d} test precision: '.format(epoch), 'test recall: '.format(epoch)]
    results = [eval_pr, eval_re, test_pr, test_re]
    for i, result in enumerate(results):
        printings = result_type[i]
        for j in result:
            printings = printings + '{:.4f}\t'.format(j)
        fid.write(printings)
        if i == 1 or i == 3:
            fid.write('\n')
    if report_ndcg:
        nd_line = 'epoch:{:2d} eval ndcg: '.format(epoch) + ''.join('{:.4f}\t'.format(j) for j in eval_nd) + '\n'
        nd_line += 'test ndcg: ' + ''.join('{:.4f}\t'.format(j) for j in test_nd) + '\n'
        fid.write(nd_line)
    fid.write('epoch:{:2d} test {}'.format(epoch, format_paper_recall_line(k_list, test_re, test_nd if report_ndcg else None)))
    fid.write('\n')
    fid.close()

def eval_logging_ctr(args, epoch, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1):
    fid = open('../result/{}_{}_ctr_{}.txt'.format(args.model_type, args.dataset, args.att), 'a')

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
'''
***********************************************
'''

def recall_k_index(k_list, k):
    if k in k_list:
        return k_list.index(k)
    return min(3, len(k_list) - 1) if len(k_list) > 0 else 0


def format_paper_recall_line(k_list, recall_vec, ndcg_vec=None):
    line = ''
    for pk in (2, 10, 50, 100):
        if pk in k_list:
            i = k_list.index(pk)
            line += 'R@{}:{:.4f}\t'.format(pk, recall_vec[i])
            if ndcg_vec is not None:
                line += 'N@{}:{:.4f}\t'.format(pk, ndcg_vec[i])
    return line.rstrip() if line else '(add K to topk_k_list for paper R@2/10/50/100)\t'


def format_paper_recall_block(k_list, recall_runs, precision_runs, ndcg_runs=None):
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
        if ndcg_runs is not None:
            nm = np.mean(ndcg_runs[:, i])
            ns = np.std(ndcg_runs[:, i])
            line += '  ndcg: {:.4f}+-{:.4f}'.format(nm, ns)
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


def get_feed_dict(args, model, data, start, end, flag=True):
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
    return feed_dict


def ctr_eval(args, sess, model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1 = model.eval(sess, get_feed_dict(args, model, data, start, start + batch_size, flag=False))
        auc_list.append(auc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list))


def topk_eval(args, sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size],
                                                    model.node_dropout: [0.]*len(eval(args.layer_size)),
                                                    model.mess_dropout: [0.]*len(eval(args.layer_size))})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start),
                       model.node_dropout: [0.]*len(eval(args.layer_size)),
                       model.mess_dropout: [0.]*len(eval(args.layer_size))})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        m = ranking_metrics_at_ks(item_sorted, test_record[user], k_list)
        for idx, k in enumerate(k_list):
            precision_list[k].append(m['precision'][idx])
            recall_list[k].append(m['recall'][idx])
            ndcg_list[k].append(m['ndcg'][idx])

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]

    return precision, recall, ndcg


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
