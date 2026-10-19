# 02_product_screens

تاریخ استخراج: 2026-09-21 07:04

غربالگری توجه حاصل‌ضربی (`--att product`)؛ پروتکل ۱۰۰ کاربر، ۵ اجرا.

## بوک کراسینگ — همجوشی fixed / gate / residual

| بازو | AUC | F1 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| fixed | 0.7148 ± 0.0005 (n=5) | 0.6561 ± 0.0010 (n=5) | 0.0662 ± 0.0203 (n=5) | 0.0589 ± 0.0160 (n=5) |
| gate | 0.7142 ± 0.0004 (n=5) | 0.6549 ± 0.0008 (n=5) | 0.0660 ± 0.0202 (n=5) | 0.0566 ± 0.0166 (n=5) |
| residual | 0.7146 ± 0.0008 (n=5) | 0.6564 ± 0.0020 (n=5) | 0.0713 ± 0.0251 (n=5) | 0.0612 ± 0.0204 (n=5) |

### آزمون معناداری residual در برابر fixed

```
Paired significance test -- dataset: book
proposed = KGCN_LightGCN (product_v3_book_res_residual_product)   base = KGCN_LightGCN (product_v3_book_res_fixed_product)
paired t-test (scipy.ttest_rel), alternative='two-sided'   * p<.05  ** p<.01  *** p<.001
runs paired = 5

metric      proposed            base                delta       t         p         sig   
AUC         0.7146+-0.0008      0.7148+-0.0005      -0.0003     -0.817    0.4597    ns    
F1          0.6564+-0.0020      0.6561+-0.0010      +0.0003     0.450     0.6761    ns    
Recall@10   0.0713+-0.0251      0.0662+-0.0203      +0.0051     1.055     0.3510    ns    
NDCG@10     0.0612+-0.0204      0.0589+-0.0160      +0.0023     0.794     0.4717    ns
```

## لست‌اف‌ام — hop1 در برابر hop2

| بازو | AUC | F1 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| n_iter=1 | 0.8301 ± 0.0010 (n=5) | 0.7512 ± 0.0026 (n=5) | 0.1631 ± 0.0257 (n=5) | 0.1165 ± 0.0105 (n=5) |
| n_iter=2 | 0.4999 ± 0.0068 (n=5) | 0.4976 ± 0.0043 (n=5) | 0.0046 ± 0.0027 (n=5) | 0.0026 ± 0.0015 (n=5) |

### آزمون معناداری hop2 در برابر hop1

```
Paired significance test -- dataset: music
proposed = KGCN_LightGCN (product_v3_music_hop2_product)   base = KGCN_LightGCN (product_v3_music_hop1_product)
paired t-test (scipy.ttest_rel), alternative='two-sided'   * p<.05  ** p<.01  *** p<.001
runs paired = 5

metric      proposed            base                delta       t         p         sig   
AUC         0.4999+-0.0068      0.8301+-0.0010      -0.3302     -85.698   0.0000    ***   
F1          0.4976+-0.0043      0.7512+-0.0026      -0.2537     -106.369  0.0000    ***   
Recall@10   0.0046+-0.0027      0.1631+-0.0257      -0.1586     -11.571   0.0003    ***   
NDCG@10     0.0026+-0.0015      0.1165+-0.0105      -0.1139     -21.419   0.0000    ***
```

