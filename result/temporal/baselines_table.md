| method | dataset | protocol | fusion | baseline | avg_auc | avg_ndcg10 | final_ndcg10 | total_sec | avg_sec_per_window | peak_train_rows | n_windows_eval |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Experience Replay | ml1m | incremental | fixed | replay | 0.5748 | 0.1227 | 0.1823 | 66.3 | 16.6 | 301510 | 4 |
| OL-LGKAT (forgetting gate) | ml1m | incremental | forgetting_gate | none | 0.5607 | 0.1082 | 0.1453 | 62.8 | 15.7 | 150755 | 4 |
| Fine-Tune | ml1m | incremental | fixed | none | 0.5601 | 0.1032 | 0.1527 | 63.1 | 15.8 | 150755 | 4 |
| EWC | ml1m | incremental | fixed | ewc | 0.5601 | 0.1032 | 0.1527 | 76.5 | 19.1 | 150755 | 4 |
| Full-Retrain (batch) | ml1m | batch | fixed | none | 0.5536 | 0.1005 | 0.1595 | 66.7 | 16.7 | 603020 | 4 |
