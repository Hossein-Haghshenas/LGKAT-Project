"""
Unit tests for src/evaluation.py (OL-LGKAT Phase 1 acceptance criterion).

NDCG@K is checked against hand-computed values on controlled toy examples:
  * exact-by-inspection cases (perfect ranking, single relevant item, empty set), and
  * a multi-item case whose DCG/IDCG are derived inline (independent of the module).

Runnable two ways:
    conda run -n lgkat python tests/test_evaluation.py     # plain (no pytest needed)
    pytest tests/test_evaluation.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from evaluation import ndcg_at_k, precision_at_k, recall_at_k, ranking_metrics_at_ks  # noqa: E402

TOL = 1e-9


def _approx(a, b, tol=TOL):
    return abs(a - b) <= tol


def test_ndcg_exact_by_inspection():
    # Perfect ranking: all relevant items at the very top -> NDCG = 1.0
    assert _approx(ndcg_at_k([1, 2, 3, 4], {1, 2, 3}, 3), 1.0)
    assert _approx(ndcg_at_k([1, 2, 3, 4], {1, 2, 3}, 4), 1.0)
    # Single relevant item ranked first -> 1.0 for any k >= 1
    assert _approx(ndcg_at_k([9, 8, 7], {9}, 1), 1.0)
    assert _approx(ndcg_at_k([9, 8, 7], {9}, 3), 1.0)
    # Single relevant item at rank 2 -> DCG = 1/log2(3), IDCG = 1/log2(2) = 1
    assert _approx(ndcg_at_k([9, 8, 7], {8}, 3), 1.0 / math.log2(3))
    # No relevant items -> 0.0 ; k <= 0 -> 0.0
    assert _approx(ndcg_at_k([1, 2, 3], set(), 3), 0.0)
    assert _approx(ndcg_at_k([1, 2, 3], {1}, 0), 0.0)


def test_ndcg_toy_multi_item():
    ranked = [10, 20, 30, 40, 50]
    relevant = {20, 40, 50}  # relevant items land at ranks 2, 4, 5  => gains [0,1,0,1,1]

    # --- Hand derivation, independent of the module under test ---
    # DCG@3  = 0/log2(2) + 1/log2(3) + 0/log2(4)
    dcg3 = 1.0 / math.log2(3)
    # IDCG@3 = ideal top-3 (3 relevant available) = 1/log2(2) + 1/log2(3) + 1/log2(4)
    idcg3 = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    expected_ndcg3 = dcg3 / idcg3
    # DCG@5  = 1/log2(3) + 1/log2(5) + 1/log2(6) ; IDCG@5 == IDCG@3 (only 3 relevant)
    dcg5 = 1.0 / math.log2(3) + 1.0 / math.log2(5) + 1.0 / math.log2(6)
    expected_ndcg5 = dcg5 / idcg3

    assert _approx(ndcg_at_k(ranked, relevant, 3), expected_ndcg3)
    assert _approx(ndcg_at_k(ranked, relevant, 5), expected_ndcg5)

    # Coarse literal pins (guard against silent formula drift)
    assert round(ndcg_at_k(ranked, relevant, 3), 4) == 0.2961
    assert round(ndcg_at_k(ranked, relevant, 5), 4) == 0.6797


def test_precision_recall_toy():
    ranked = [10, 20, 30, 40, 50]
    relevant = {20, 40, 50}
    assert _approx(precision_at_k(ranked, relevant, 3), 1.0 / 3.0)
    assert _approx(recall_at_k(ranked, relevant, 3), 1.0 / 3.0)
    assert _approx(precision_at_k(ranked, relevant, 5), 0.6)
    assert _approx(recall_at_k(ranked, relevant, 5), 1.0)


def test_ranking_bundle_matches_individual():
    ranked = [10, 20, 30, 40, 50]
    relevant = {20, 40, 50}
    k_list = [2, 3, 5]
    m = ranking_metrics_at_ks(ranked, relevant, k_list)
    for idx, k in enumerate(k_list):
        assert _approx(m['precision'][idx], precision_at_k(ranked, relevant, k))
        assert _approx(m['recall'][idx], recall_at_k(ranked, relevant, k))
        assert _approx(m['ndcg'][idx], ndcg_at_k(ranked, relevant, k))


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print('PASS  ' + t.__name__)
        except AssertionError as e:
            failed += 1
            print('FAIL  ' + t.__name__ + '  ' + repr(e))
    if failed:
        print('\n%d of %d test(s) failed' % (failed, len(tests)))
        sys.exit(1)
    print('\nAll %d tests passed' % len(tests))
