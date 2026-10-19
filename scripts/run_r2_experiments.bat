@echo off
rem Reviewer-2 follow-up experiments: the Last.FM ablation, then the Book-Crossing CL arm.
rem Runs in the window "R2 experiments"; close it to stop. A relaunch skips finished arms.
rem Log: result\r2_all.log
cd /d D:\projects\current-git-project\LGKAT-Project
start "R2 experiments" cmd /c "C:\Users\asus\.conda\envs\lgkat\python.exe -u scripts\run_r2_experiments.py --queue all >> result\r2_all.log 2>&1"
