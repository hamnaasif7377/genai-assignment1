# GenAI Assignment 1 - Pet Image Restoration + Face-to-Sketch GAN

Implementation of four generative AI systems: a universal denoising autoencoder,
a hard-routed specialist system, a soft mixture-of-experts model, and a
conditional GAN for face-to-sketch generation.

## Repository Structure
repo/
├── src/
│ ├── data/
│ │ ├── corruptions.py # Corruption functions (salt-pepper, blur, occlusion)
│ │ ├── datasets.py # PyTorch Dataset classes
│ │ └── manifests.py # Deterministic val/test manifest generation
│ ├── models/
│ │ └── task1_autoencoder.py # Universal autoencoder architecture
│ ├── training/
│ │ ├── losses.py # L1 + SSIM combined loss
│ │ └── train_task1.py # Optuna search + final training script
│ ├── evaluation/ # Evaluation scripts (per-corruption/severity results)
│ └── onnx_export/ # ONNX export + verification scripts
├── configs/
│ └── task1_best_params.json # Optuna's best hyperparameters for Task 1
├── scripts/ # Colab setup helpers
├── frontend/ # React + Tailwind application
└── requirements.txt

## Setup

### Local (Windows, for app development)
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Google Colab (for training, GPU required)
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/hamnaasif7377/genai-assignment1.git /content/repo
%cd /content/repo
!pip install -q -r requirements.txt

import sys
sys.path.insert(0, '/content/repo')
```

## Dataset

Oxford-IIIT Pet Dataset (Tasks 1-3), downloaded automatically via torchvision:
```python
from torchvision.datasets import OxfordIIITPet
OxfordIIITPet(root='/content/data', split='trainval', download=True)
OxfordIIITPet(root='/content/data', split='test', download=True)
```

Official 80/20 train/val split with seed=42, applied consistently across Tasks 1-3.

## Task 1: Universal Multi-Corruption Denoising Autoencoder

### Run the Optuna hyperparameter search
```bash
python -m src.training.train_task1 --mode search --n_trials 30 \
    --data_root /content/data \
    --manifest_root /content/drive/MyDrive/GenAI_A1/manifests \
    --optuna_root /content/drive/MyDrive/GenAI_A1/optuna
```

### Train the final model with the best found hyperparameters
```bash
python -m src.training.train_task1 --mode final --epochs 60 \
    --data_root /content/data \
    --manifest_root /content/drive/MyDrive/GenAI_A1/manifests \
    --checkpoint_root /content/drive/MyDrive/GenAI_A1/checkpoints
```

Resume an interrupted training run:
```bash
python -m src.training.train_task1 --mode final --epochs 60 --resume \
    --checkpoint_root /content/drive/MyDrive/GenAI_A1/checkpoints
```

### Results (current best)
- Optuna search: 30 trials, best val_loss = 0.1341
- Final training: 60 epochs, best val_loss = 0.1059
- Best hyperparameters: see `configs/task1_best_params.json`

## Tasks 2-4

(To be added as development progresses.)

## AI-Use Appendix

This project was developed with assistance from Claude (Anthropic) for:
- Environment setup and debugging (PowerShell, Docker, Colab configuration)
- Code structure and implementation of data pipeline, model architecture, training loop
- Debugging runtime errors (variable scope issues, DataLoader collation, MLflow backend)

All generated code was tested by running it and inspecting outputs (loss curves,
visual reconstructions, manifest correctness) before being accepted into the codebase.
