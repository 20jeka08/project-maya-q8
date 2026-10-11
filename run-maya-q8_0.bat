@echo off
rem starts the Project Maya dashboard and API (written by maya.py; START-MAYA.bat does the same)
cd /d "E:\progs\Maya\project-maya" || exit /b 1
E:\progs\Maya\project-maya\.venv\Scripts\python.exe E:\progs\Maya\project-maya\maya.py --config E:\progs\Maya\project-maya\maya-q8_0.json --port 8080 %*
