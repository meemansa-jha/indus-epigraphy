@echo off
REM One-time setup on Windows. Run from inside the unzipped indus_demo folder:  demo\setup_windows.bat
cd /d "%~dp0\.."
py -3 -m venv .venv || python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r demo\requirements-demo.txt
python -c "import torch, ultralytics, cv2, sklearn; print('setup OK - torch', torch.__version__)"
