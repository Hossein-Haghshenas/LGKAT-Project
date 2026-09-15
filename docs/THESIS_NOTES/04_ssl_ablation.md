# 04_ssl_ablation

تاریخ استخراج: 2026-09-15 21:43

حذف اجزای یادگیری خودنظارتی با سه اجرا. بازوی بدون CL در برابر بازوی با CL.
توجه: برچسب قدیمی `..._005` یعنی `cl_lambda=0.005` نه ۰٫۰۵.

| مجموعه‌داده | بازو | AUC | F1 | Recall@10 |
|---|---|---|---|---|
| book | بدون CL | 0.7142 ± 0.0002 (n=3) | 0.6551 ± 0.0014 (n=3) | 0.0669 ± 0.0161 (n=3) |
| book | CL λ=0.005 | 0.6077 ± 0.0529 (n=3) | 0.5965 ± 0.0326 (n=3) | 0.0425 ± 0.0132 (n=3) |
| movie | بدون CL | 0.9813 ± 0.0002 (n=3) | 0.9422 ± 0.0004 (n=3) | 0.2075 ± 0.0314 (n=3) |
| movie | CL λ=0.005 | 0.9811 ± 0.0001 (n=3) | 0.9427 ± 0.0008 (n=3) | 0.2020 ± 0.0283 (n=3) |
| restaurant | بدون CL | 0.8729 ± 0.0003 (n=3) | 0.7847 ± 0.0004 (n=3) | 0.1234 ± 0.0138 (n=3) |
| restaurant | CL λ=0.02 sample=256 | 0.8636 ± 0.0010 (n=3) | 0.7785 ± 0.0013 (n=3) | 0.0947 ± 0.0142 (n=3) |

## معناداری book (CL در برابر بدون CL)

```
Paired significance test -- dataset: book
proposed = KGCN_LightGCN (product_ssl_ablate_3run_book_cl005)   base = KGCN_LightGCN (product_ssl_ablate_3run_book_nocl)
paired t-test (scipy.ttest_rel), alternative='two-sided'   * p<.05  ** p<.01  *** p<.001
runs paired = 3

metric      proposed            base                delta       t         p         sig   
AUC         0.6077+-0.0529      0.7142+-0.0002      -0.1064     -2.849    0.1043    ns    
F1          0.5965+-0.0326      0.6551+-0.0014      -0.0586     -2.549    0.1256    ns    
Recall@10   0.0425+-0.0132      0.0669+-0.0161      -0.0244     -1.174    0.3613    ns    
NDCG@10     0.0397+-0.0082      0.0546+-0.0161      -0.0149     -0.867    0.4772    ns
```

## معناداری movie (CL در برابر بدون CL)

```
Paired significance test -- dataset: movie
proposed = KGCN_LightGCN (product_ssl_ablate_3run_movie_cl005)   base = KGCN_LightGCN (product_ssl_ablate_3run_movie_nocl)
paired t-test (scipy.ttest_rel), alternative='two-sided'   * p<.05  ** p<.01  *** p<.001
runs paired = 3

metric      proposed            base                delta       t         p         sig   
AUC         0.9811+-0.0001      0.9813+-0.0002      -0.0002     -0.992    0.4258    ns    
F1          0.9427+-0.0008      0.9422+-0.0004      +0.0004     0.598     0.6104    ns    
Recall@10   0.2020+-0.0283      0.2075+-0.0314      -0.0055     -0.315    0.7827    ns    
NDCG@10     0.2136+-0.0358      0.2055+-0.0281      +0.0081     0.658     0.5782    ns
```

## معناداری restaurant (CL در برابر بدون CL)

```
Paired significance test -- dataset: restaurant
proposed = KGCN_LightGCN (product_ssl_ablate_3run_restaurant_cl256)   base = KGCN_LightGCN (product_ssl_ablate_3run_restaurant_nocl)
paired t-test (scipy.ttest_rel), alternative='two-sided'   * p<.05  ** p<.01  *** p<.001
runs paired = 3

metric      proposed            base                delta       t         p         sig   
AUC         0.8636+-0.0010      0.8729+-0.0003      -0.0093     -10.106   0.0097    **    
F1          0.7785+-0.0013      0.7847+-0.0004      -0.0062     -5.463    0.0319    *     
Recall@10   0.0947+-0.0142      0.1234+-0.0138      -0.0286     -2.733    0.1119    ns    
NDCG@10     0.0639+-0.0215      0.0764+-0.0097      -0.0125     -1.077    0.3941    ns
```

اگر دلتا منفی یا ns باشد، همان پاسخ روشن به «چرا یادگیری تقابلی فقط روی لست‌اف‌ام روشن است» است.

