@echo off
rem Reviewer-4 experiments for the LGKAT-SSL article (queue r4 in run_r2_experiments.py):
rem   1. Last.FM: gate values and per-pair test scores of the final model and of LGKAT, 3 runs each
rem   2. noise term with fixed fusion and with the gate on Book-Crossing, Dianping and MovieLens, 3 runs each
rem Safe to start while the fixed alpha + noise arm still runs: it waits until that ends.
rem About 10 hours in the window "R4 experiments". Close that window to stop it; a restart skips finished arms.
rem Log: result\r4.log
cd /d D:\projects\current-git-project\LGKAT-Project
start "R4 experiments" cmd /c "C:\Users\asus\.conda\envs\lgkat\python.exe -u scripts\run_r2_experiments.py --queue r4 --wait 1 >> result\r4.log 2>&1"
