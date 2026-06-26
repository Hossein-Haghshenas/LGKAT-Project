"""
Unit tests for the per-item StabilityTracker (OL-LGKAT Phase 4). Pure NumPy, fast.

    conda run -n lgkat python tests/test_stability.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from stability import StabilityTracker  # noqa: E402


def test_zero_before_any_window():
    tr = StabilityTracker(5, 4)
    f = tr.features(0)
    assert f.shape == (5, 4)
    assert np.allclose(f, 0.0)


def test_seen_items_get_age_and_freq():
    tr = StabilityTracker(5, 4)
    tr.update(0, [0, 0, 1])          # items 0,1 active in window 0
    f = tr.features(1)               # entering window 1; history = window 0
    # age = (1-0)/1 = 1 ; freq = 1/1 = 1 for seen items
    assert f[0, 0] == 1.0 and f[0, 1] == 1.0
    assert f[1, 0] == 1.0 and f[1, 1] == 1.0
    # unseen items stay all-zero (=> gate falls back to identity-init alpha)
    assert f[3].sum() == 0.0
    assert f[4].sum() == 0.0
    # all features bounded in [0,1]
    assert f.min() >= 0.0 and f.max() <= 1.0


def test_drift_only_for_moved_items():
    tr = StabilityTracker(3, 4)
    emb0 = np.zeros((3, 2), dtype=np.float32)
    tr.update(0, [0, 1], emb0)                       # first embeddings -> no drift yet
    emb1 = np.array([[1, 0], [0, 0], [0, 0]], dtype=np.float32)
    tr.update(1, [0], emb1)                          # only item 0 moved (by 1.0)
    assert tr.drift[0] > 0.0
    assert tr.drift[1] == 0.0 and tr.drift[2] == 0.0
    # lower drift -> higher stability feature
    f = tr.features(2)
    assert f[1, 3] >= f[0, 3]


def test_save_load_roundtrip():
    tr = StabilityTracker(4, 4)
    tr.update(0, [0, 1], np.ones((4, 3), dtype=np.float32))
    tr.update(1, [2], np.full((4, 3), 2.0, dtype=np.float32))
    p = os.path.join(tempfile.mkdtemp(), 's.npz')
    tr.save_npz(p)
    tr2 = StabilityTracker.load_npz(p)
    assert np.array_equal(tr.first_seen, tr2.first_seen)
    assert np.array_equal(tr.last_seen, tr2.last_seen)
    assert np.array_equal(tr.count, tr2.count)
    assert np.allclose(tr.drift, tr2.drift)
    assert np.allclose(tr.features(2), tr2.features(2))


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
