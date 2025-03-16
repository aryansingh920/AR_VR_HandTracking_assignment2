#!/bin/zsh

pip install -r requirements.txt

python3 main.py --mode opengl
python3 prediction.py
