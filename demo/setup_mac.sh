#!/usr/bin/env bash
# One-time setup on macOS (or Linux). Run from inside the unzipped indus_demo folder:  bash demo/setup_mac.sh
set -e
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r demo/requirements-demo.txt
python -c "import torch, ultralytics, cv2, sklearn; print('setup OK - torch', torch.__version__)"
