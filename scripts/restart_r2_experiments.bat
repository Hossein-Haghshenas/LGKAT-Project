@echo off
rem Stops the running Reviewer-2 experiments, writes a short GPU/power report, restarts them.
rem Double-click this one file: the old slow run is ended and the new settings take effect.
rem Report: result\r2_diag.txt   Log: result\r2_all.log
cd /d D:\projects\current-git-project\LGKAT-Project

echo Stopping the old experiment processes ...
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'run_r2_experiments.py|--exp_tag r2_' } | ForEach-Object { Write-Host ('stopping PID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force }"
timeout /t 5 /nobreak >nul

echo Writing result\r2_diag.txt ...
> result\r2_diag.txt echo === date
date /t >> result\r2_diag.txt
time /t >> result\r2_diag.txt
echo === active power scheme >> result\r2_diag.txt
powercfg /getactivescheme >> result\r2_diag.txt 2>&1
echo === gpu state >> result\r2_diag.txt
nvidia-smi --query-gpu=name,pstate,clocks.sm,clocks.max.sm,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks_throttle_reasons.active --format=csv >> result\r2_diag.txt 2>&1
echo === processes on the gpu >> result\r2_diag.txt
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >> result\r2_diag.txt 2>&1
echo === python processes >> result\r2_diag.txt
powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU,PriorityClass,WorkingSet | Format-Table -AutoSize | Out-String -Width 200" >> result\r2_diag.txt 2>&1

echo Restarting the experiments in the window "R2 experiments" ...
start "R2 experiments" cmd /c "C:\Users\asus\.conda\envs\lgkat\python.exe -u scripts\run_r2_experiments.py --queue all >> result\r2_all.log 2>&1"
echo Done. This window can be closed; keep "R2 experiments" open.
timeout /t 10 /nobreak >nul
