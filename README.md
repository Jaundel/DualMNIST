# DualMNIST Quick Start

## Setup

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## NumPy NN

python src_numpy/train.py

## PyTorch CNN

python src_pytorch/train.py
