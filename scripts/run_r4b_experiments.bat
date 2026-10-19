@echo off
rem Reviewer-4 re-run (queue r4b in run_r2_experiments.py): the noise term with fixed fusion and with
rem the gate on Dianping and MovieLens, 3 runs each. The first r4 runs of these four arms stopped on NaN;
rem src\model.py now computes the noise scale with a zero-safe norm (backup in _backup_r4b).
rem About 7 hours in the window "R4b experiments". Close that window to stop it; a restart skips finished arms.
rem Log: result\r4b.log
cd /d D:\projects\current-git-project\LGKAT-Project
start "R4b experiments" cmd /c "C:\Users\asus\.conda\envs\lgkat\python.exe -u scripts\run_r2_experiments.py --queue r4b --wait 1 >> result\r4b.log 2>&1"
