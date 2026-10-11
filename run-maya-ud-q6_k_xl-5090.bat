@echo off
rem runs GLM-5.3-Flash UD-Q6_K_XL on the RTX 5090 alone (CUDA device 1) - no layer split
cd /d "E:\progs\Maya\project-maya" || exit /b 1
E:\progs\Maya\project-maya\.venv\Scripts\python.exe E:\progs\Maya\project-maya\maya.py --config E:\progs\Maya\project-maya\maya-ud-q6_k_xl-5090.json --port 8080 %*
