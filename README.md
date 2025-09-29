# MNIST Neural Network

Quick project comparing two ways to classify MNIST digits:
- A NumPy neural net to see the applied mathematics behind forward/backprop.
- A PyTorch CNN for a standard deep-learning baseline.

## Overview
- **NumPy:** manual forward/backprop, gradient descent, sigmoid activations  
- **PyTorch:** conv layers, ReLU, max-pool, fully connected layers  
- Wanted to see differences in accuracy, training stability, and effort

## Layout
```text
├── data/                 # MNIST dataset (auto-downloads on first run)
├── src_numpy/            # NumPy model + training script
└── src_pytorch/          # PyTorch CNN + training script
```
## Setup
- pip install -r requirements.txt

## Run

### NumPy version
python src_numpy/train.py

### PyTorch verersion
python src_pytorch/train.py
