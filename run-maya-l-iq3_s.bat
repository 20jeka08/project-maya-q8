@echo off
rem starts the Project Maya dashboard and API with the Maya-L IQ3_S model (E:\progs\Maya-data\models\Maya-L)
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'E:\\progs\\[Ss]trata' -and $_.Name -match '^(strata|strata-vision|python|pythonw|cmd)\.exe$' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }"
cd /d "E:\progs\Maya\project-maya" || exit /b 1
E:\progs\Maya\project-maya\.venv\Scripts\python.exe E:\progs\Maya\project-maya\maya.py --config E:\progs\Maya\project-maya\maya-l-iq3_s.json --port 8080 %*
