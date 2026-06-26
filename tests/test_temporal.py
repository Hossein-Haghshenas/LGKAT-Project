"""
Unit tests for the temporal/streaming protocol (OL-LGKAT Phase 3 acceptance criterion:
"zero future-to-past data leakage").

Checks on `data_loader.temporal_windows`:
  * windows partition the data exactly (no loss, no duplication),
  * windows are time-ordered: max(ts of window i) <= min(ts of window i+1),
  * the causal training set for evaluating window t+1 (= windows 0..t) contains NO
    interaction with a timestamp later than the evaluation window  => no leakage,
  * a dataset without a timestamp column is rejected.

Runnable directly or via pytest:
    conda run -n lgkat python tests/test_temporal.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from data_loader import temporal_windows  # noqa: E402


def _toy(n=100, seed=0):
    """n interactions with strictly varied timestamps, deliberately shuffled."""
    rng = np.random.RandomState(seed)
    users = rng.randint(0, 10, size=n)
    items = rng.randint(0, 20, size=n)
    labels = rng.randint(0, 2, size=n)
    ts = rng.permutation(1000 + np.arange(n) * 7)  # distinct, out of order
    return np.stack([users, items, labels, ts], axis=1).astype(np.int64)


def test_partition_is_exact():
    data = _toy(100)
    wins = temporal_windows(data, 7)
    total = sum(w.shape[0] for w in wins)
    assert total == data.shape[0]
    # multiset of timestamps preserved
    got = np.sort(np.concatenate([w[:, 3] for w in wins]))
    assert np.array_equal(got, np.sort(data[:, 3]))


def test_windows_are_time_ordered():
    data = _toy(100)
    wins = temporal_windows(data, 7)
    for i in range(len(wins) - 1):
        assert wins[i][:, 3].max() <= wins[i + 1][:, 3].min()


def test_no_future_to_past_leakage():
    data = _toy(120)
    wins = temporal_windows(data, 8)
    for t in range(len(wins) - 1):
        train = np.concatenate(wins[: t + 1], axis=0)   # causal training set (windows 0..t)
        eval_win = wins[t + 1]                            # future evaluation window
        # no training interaction occurs after the start of the evaluation window
        assert train[:, 3].max() <= eval_win[:, 3].min()


def test_rejects_dataset_without_timestamps():
    data3 = np.zeros((10, 3), dtype=np.int64)  # user item label, no timestamp
    try:
        temporal_windows(data3, 5)
    except ValueError:
        return
    raise AssertionError('temporal_windows should reject a 3-column (no-timestamp) dataset')


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
