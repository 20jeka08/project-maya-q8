@echo off
rem runs GLM-5.3 UD-Q4_K_XL (glm-dsa) with Maya's dashboard and API: dense weights on the RTX 5090, routed experts on the CPU
rem first stops any running Maya / Strata engine (GLM-5.3 Flash included): both cannot share the GPU and the RAM
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'E:\\progs\\(Maya\\project-maya|project-maya|[Ss]trata)' -and $_.Name -match '^(strata|strata-vision|glm_dsa_run|python|pythonw)\.exe$' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }"
cd /d "E:\progs\project-maya" || exit /b 1
E:\progs\project-maya\.venv\Scripts\python.exe E:\progs\project-maya\maya.py --config E:\progs\project-maya\maya-glm53.json --port 8080 %*
