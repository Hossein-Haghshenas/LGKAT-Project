@echo off
rem Extra Last.FM arm for the Reviewer-2 ablation: fixed alpha + noise term, no cross-view term.
rem Safe to start while the R2 experiments still run: it waits until they finish, then runs
rem 5 runs (about 1.7 h) in the window "R2 extra arm". Close that window to stop it.
rem Log: result\r2_extra.log
cd /d D:\projects\current-git-project\LGKAT-Project
start "R2 extra arm" cmd /c "C:\Users\asus\.conda\envs\lgkat\python.exe -u scripts\run_r2_experiments.py --queue extra --wait 1 >> result\r2_extra.log 2>&1"
