# Reviewer-2 follow-up experiments for the LGKAT-SSL article.
#   music: Last.FM component ablation of paper_ssl_final_5run on the current code,
#          5 runs per arm: full, no SimGCL, no CL, fixed fusion, gate only, fixed only.
#   book : gate + CL (lambda 0.005), same flags as ssl_ablate_3run_book_cl005 but early
#          stopping off (all 39 epochs) and 5 runs, so slow-start runs are not cut early.
# Early stopping is off on Last.FM too (see music_arms): with patience 12 on eval R@10 a run
# can be killed during the plateau, before the escape at epoch ~30, purely on noise.
# Both queues at once on one GPU made Last.FM ~6x slower, so 'all' runs music, then book.
# A minimized run was still ~4x slower (95 vs 23 s/epoch), the pattern of Windows power
# throttling of background processes, so the runner opts itself and each arm out of it.
# scripts/run_r2_experiments.bat starts --queue all; one queue by hand, from repo root:
#   C:\Users\asus\.conda\envs\lgkat\python.exe -u scripts\run_r2_experiments.py --queue book
from __future__ import print_function

import argparse
import ctypes
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_closure_under10h import (RESULT, SRC, _cmd, _env, book_champ, movie_champ, music_ssl,
                                 overlay, restaurant_champ, runs_done)


def book_arms():
    cl = overlay(book_champ(), runs='5', fusion='gate', use_cl='1', cl_lambda='0.005',
                 cl_tau='0.2', use_simgcl='0', early_stop='0')
    return [('r2_book_cl005_noes_5run', cl)]


def music_full():
    return overlay(music_ssl(), runs='5', early_stop='0')


def extra_arms():
    # Missing cell of the Last.FM ablation: fixed alpha + noise term, no cross-view term.
    # gate_only (0.8078) is below fixed_only (0.8189) while gate + noise (0.8347) is the best
    # arm, so this arm shows whether the gate adds anything once the noise term is on.
    return [('r2_music_fixed_noise_5run', overlay(music_full(), fusion='fixed', use_cl='0'))]


def r4_arms(champ, cl_sample_size='0'):
    # Reviewer 4: the noise term on the other three datasets, with and without the gate, on
    # each dataset's final configuration and Last.FM's untuned lambda 0.02, eps 0.05, tau 0.2.
    # 3 runs per arm. Early stopping keeps each dataset's patience but never before epoch 10,
    # so a slow start is not cut short (the Book cross-view arm lost two runs that way).
    base = overlay(champ(), runs='3', use_cl='0', use_simgcl='1', simgcl_lambda='0.02',
                   simgcl_eps='0.05', cl_tau='0.2', cl_sample_size=cl_sample_size,
                   early_stop_min_epoch='10')
    name = dict(base)['dataset']
    return [('r4_%s_fixed_noise_3run' % name, overlay(base, fusion='fixed')),
            ('r4_%s_gate_noise_3run' % name, overlay(base, fusion='gate'))]


def r4_music_arms():
    # Reviewer 4: per-pair test scores and gate values (--pair_dump 1) for the final Last.FM
    # model (gate + noise) and for LGKAT on the same schedule, 3 runs each.
    final = overlay(music_full(), use_cl='0', runs='3', pair_dump='1')
    return [('r4_music_final_dump_3run', final),
            ('r4_music_fixed_dump_3run', overlay(final, fusion='fixed', use_simgcl='0'))]


def music_arms():
    # early_stop 0: all five published runs went the full 39 epochs (best epoch 34-36), so
    # patience never bound there, but it is luck. Early stopping follows eval R@10 with a
    # strict '>', and R@10 is near zero and noisy until the escape at epoch ~30: a high early
    # value that stands for 12 epochs ends the run. r2_music_full_5run run 2 died that way at
    # epoch 13 with AUC 0.518 (best R@10 0.0100 at epoch 1). Checkpoint choice is unchanged.
    full = music_full()
    return [
        ('r2_music_full_5run', full),
        ('r2_music_nosimgcl_5run', overlay(full, use_simgcl='0')),
        ('r2_music_nocl_5run', overlay(full, use_cl='0')),
        ('r2_music_fixed_ssl_5run', overlay(full, fusion='fixed')),
        ('r2_music_gate_only_5run', overlay(full, use_cl='0', use_simgcl='0')),
        ('r2_music_fixed_only_5run', overlay(full, fusion='fixed', use_cl='0', use_simgcl='0')),
    ]


class _Throttling(ctypes.Structure):
    _fields_ = [('Version', ctypes.c_ulong), ('ControlMask', ctypes.c_ulong),
                ('StateMask', ctypes.c_ulong)]


def no_throttling(handle=None):
    """Exempt a process from Windows power throttling (EcoQoS) of background work."""
    if os.name != 'nt':
        return
    try:
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                              ctypes.POINTER(_Throttling), ctypes.c_ulong]
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        state = _Throttling(1, 0x1, 0x0)  # version 1, EXECUTION_SPEED controlled, throttling off
        h = k32.GetCurrentProcess() if handle is None else handle
        # 4 = ProcessPowerThrottling
        if not k32.SetProcessInformation(h, 4, ctypes.byref(state), ctypes.sizeof(state)):
            print('power throttling opt-out failed: %d' % ctypes.get_last_error(), flush=True)
    except Exception as e:
        print('power throttling opt-out skipped: %s' % e, flush=True)


def run_arm(name, cmd, skip):
    """run_closure_under10h.run_arm with the child exempt from power throttling."""
    print('=' * 80, flush=True)
    print('ARM %s' % name, flush=True)
    if skip:
        print('SKIP already complete: %s' % name, flush=True)
        print('ARM_DONE %s' % name, flush=True)
        return 0
    print(' '.join(cmd), flush=True)
    print('=' * 80, flush=True)
    proc = subprocess.Popen(cmd, env=_env(), cwd=SRC)
    no_throttling(int(proc._handle) if os.name == 'nt' else None)
    rc = proc.wait()
    if rc != 0:
        print('ARM_FAILED %s rc=%s' % (name, rc), flush=True)
        return rc
    print('ARM_DONE %s' % name, flush=True)
    return 0


QUEUES = {'music': [('music', music_arms)], 'book': [('book', book_arms)],
          'all': [('music', music_arms), ('book', book_arms)],
          'extra': [('music', extra_arms)],
          'r4': [('music', r4_music_arms), ('book', lambda: r4_arms(book_champ)),
                 ('restaurant', lambda: r4_arms(restaurant_champ, '256')),
                 ('movie', lambda: r4_arms(movie_champ))]}


def other_runners():
    """PIDs of other python processes running these experiments (Windows)."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    try:
        out = subprocess.check_output(['powershell', '-NoProfile', '-Command', ps],
                                      universal_newlines=True)
        procs = json.loads(out) if out.strip() else []
    except Exception as e:
        print('process check skipped: %s' % e, flush=True)
        return []
    if isinstance(procs, dict):
        procs = [procs]
    return [p.get('ProcessId') for p in procs if p.get('ProcessId') != os.getpid() and
            any(s in (p.get('CommandLine') or '') for s in
                ('run_r2_experiments.py', '--exp_tag r2_', '--exp_tag r4_'))]


def park_partial(tag):
    """Rename files left by an interrupted arm so its rerun starts from clean logs."""
    stamp = time.strftime('%Y%m%d-%H%M%S')
    for path in glob.glob(os.path.join(RESULT, '*_%s.*' % tag)):
        if '.aborted-' not in path:
            os.rename(path, '%s.aborted-%s' % (path, stamp))
            print('parked %s' % os.path.basename(path), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', choices=sorted(QUEUES), default='all')
    ap.add_argument('--dry_run', type=int, default=0)
    ap.add_argument('--wait', type=int, default=0,
                    help='1: wait for running experiment processes to finish instead of refusing')
    args = ap.parse_args()
    if not args.dry_run:
        no_throttling()
        others = other_runners()
        while others and args.wait:
            print('waiting for running experiments (PIDs %s) to finish' % others, flush=True)
            time.sleep(120)
            others = other_runners()
        if others:
            print('experiment processes already running (PIDs %s); close them first' % others,
                  flush=True)
            return 1
    for dataset, arms in QUEUES[args.queue]:
        for tag, pairs in arms():
            cmd = _cmd(overlay(pairs, exp_tag=tag))
            if args.dry_run:
                print(tag, ' '.join(cmd[1:]))
                continue
            done = runs_done(dataset, 'product', tag, int(dict(pairs).get('runs', 5)))
            if not done:
                park_partial(tag)
            rc = run_arm(tag, cmd, done)
            if rc != 0:
                return rc
    print('QUEUE_DONE %s' % args.queue, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
