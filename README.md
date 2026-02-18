# Neural IK Solver

A neural network-based inverse kinematics (IK) solver for the KUKA IIWA robot arm using PyBullet and PyTorch.

## Overview

This project trains a deep neural network to predict joint angles from target end-effector positions, providing a fast alternative to traditional IK solvers. The model learns from data generated using PyBullet's built-in IK solver.

## Quick Start

### Requirements
- PyBullet
- PyTorch
- NumPy

### Usage

1. **Data Collection** (optional, ~3000 samples provided)
   ```python
   DO_DATA_COLLECTION = True
   ```

2. **Model Training**
   ```python
   DO_TRAINING = True
   ```

3. **Inference/Testing**
   ```python
   DO_INFERENCE = True
   ```

Run:
```bash
python main.py
```

## Configuration

Edit flags and hyperparameters at the top of `main.py`:
- `DO_DATA_COLLECTION`: Generate new training data
- `DO_TRAINING`: Train the model
- `DO_INFERENCE`: Run simulation with trained model
- `NUM_SAMPLES`, `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`: Training hyperparameters

## Files

- `main.py` - Main script (data collection, training, inference)
- `data/ik_data.pt` - Training dataset
- `models/ik_model.pth` - Trained model weights
