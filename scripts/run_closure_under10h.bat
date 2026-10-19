@echo off
cd /d D:\projects\current-git-project\LGKAT-Project
set PYTHONUNBUFFERED=1
set PYTHONHASHSEED=0
echo ===== %date% %time% resume --only 6 =====>> result\closure_under10h.log
C:\Users\asus\.conda\envs\lgkat\python.exe -u scripts\run_closure_under10h.py --only 6 --commit 0 >> result\closure_under10h.log 2>> result\closure_under10h.err
echo ===== %date% %time% python exit %ERRORLEVEL% =====>> result\closure_under10h.log
