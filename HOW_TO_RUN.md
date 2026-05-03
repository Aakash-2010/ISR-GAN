# How to Run SuperResolution-Master

## Project Overview

This project has **3 runnable components**:

| Component | Framework | What it does |
|-----------|-----------|--------------|
| `apps/streamlit-esrgan-demo/` | PyTorch + Streamlit | Web UI to upscale images using ESRGAN |
| `projects/torch-srgan/` | PyTorch | Train/test SRGAN from scratch |
| `projects/tf-srgan-div2k/` | TensorFlow | Train SRGAN on DIV2K dataset |

---

## Quick Command Reference

```bash
# ── Setup ──────────────────────────────────────────────────────────────────
pip install -r requirements.txt                          # install all PyTorch deps
pip install tensorflow tensorflow-datasets               # only needed for TF SRGAN

# ── Streamlit ESRGAN Demo ──────────────────────────────────────────────────
cd apps/streamlit-esrgan-demo
streamlit run app.py                                     # opens at http://localhost:8501
# (model auto-downloads from Hugging Face on first Upscale click)

# ── PyTorch SRGAN: inference (no dataset needed) ───────────────────────────
cd projects/torch-srgan
python main.py --mode test_only \
  --generator_path ./pretrained_models/SRGAN.pt \
  --LR_path /path/to/your/low-res/images

# ── PyTorch SRGAN: test with PSNR score ────────────────────────────────────
python main.py --mode test \
  --generator_path ./pretrained_models/SRGAN.pt \
  --LR_path ../DIV2K/DIV2K_train_LR_bicubic/X4 \
  --GT_path ../DIV2K/DIV2K_train_HR

# ── PyTorch SRGAN: train from scratch ─────────────────────────────────────
python main.py --mode train \
  --LR_path ../DIV2K/DIV2K_train_LR_bicubic/X4 \
  --GT_path ../DIV2K/DIV2K_train_HR

# ── PyTorch SRGAN: quick smoke-test (small epoch count) ───────────────────
python main.py --mode train \
  --pre_train_epoch 10 --fine_train_epoch 5 \
  --batch_size 4 --in_memory False \
  --LR_path ../DIV2K/DIV2K_train_LR_bicubic/X4 \
  --GT_path ../DIV2K/DIV2K_train_HR

# ── PyTorch SRGAN: resume fine-tuning from checkpoint ─────────────────────
python main.py --mode train --fine_tuning True \
  --generator_path ./model/pre_trained_model_0800.pt \
  --LR_path ../DIV2K/DIV2K_train_LR_bicubic/X4 \
  --GT_path ../DIV2K/DIV2K_train_HR

# ── TF SRGAN: one-time data prep ──────────────────────────────────────────
cd projects/tf-srgan-div2k
python create_tfrecords.py                               # downloads DIV2K + makes TFRecords

# ── TF SRGAN: train ────────────────────────────────────────────────────────
python train_srgan.py

# ── TF SRGAN: visualise results ───────────────────────────────────────────
python inference.py
```

---

## 1. Setup

### Using the project virtual environment (recommended)

```bash
# Activate the existing venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

### TensorFlow (only if using tf-srgan-div2k)

```bash
pip install tensorflow tensorflow-datasets
```

---

## 2. Streamlit ESRGAN Demo

The model (`RealESRGAN_x4.pth`) is **auto-downloaded from Hugging Face** the first time you click Upscale. No manual download needed.

### Run

```bash
cd apps/streamlit-esrgan-demo
streamlit run app.py
```

Opens at `http://localhost:8501`

### How to use

1. Upload one or more images (JPG/PNG)
2. Select **CPU** or **CUDA** — CUDA only appears if a compatible GPU is detected
3. Click **Upscale!** — first run downloads the model (~65 MB), subsequent runs are fast
4. Click **Create and Download Archive** to download results as a ZIP

### Output

- Upscaled images → `apps/streamlit-esrgan-demo/results/`
- Original inputs archived to → `apps/streamlit-esrgan-demo/old/`
- Model weights cached at → `apps/streamlit-esrgan-demo/weights/RealESRGAN_x4.pth`

---

## 3. PyTorch SRGAN (`projects/torch-srgan/`)

### Inference — no dataset needed (pretrained models included)

```bash
cd projects/torch-srgan

# Upscale images with no ground truth
python main.py --mode test_only \
  --generator_path ./pretrained_models/SRGAN.pt \
  --LR_path /path/to/your/low-res/images

# Upscale + compute PSNR (needs DIV2K)
python main.py --mode test \
  --generator_path ./pretrained_models/SRGAN.pt \
  --LR_path ../DIV2K/DIV2K_train_LR_bicubic/X4 \
  --GT_path ../DIV2K/DIV2K_train_HR
```

Results saved to `projects/torch-srgan/result/`

### Training — needs DIV2K dataset

Download DIV2K from https://data.vision.ee.ethz.ch/cvl/DIV2K/ (or Kaggle: search "DIV2K").
Place the `DIV2K/` folder **one level above** this repo.

```
SuperResolution-Master/    ← this repo
DIV2K/
  DIV2K_train_HR/
  DIV2K_train_LR_bicubic/
    X4/
```

```bash
cd projects/torch-srgan

# Full training (8000 pretrain + 4000 finetune epochs — takes hours)
python main.py --mode train \
  --LR_path ../DIV2K/DIV2K_train_LR_bicubic/X4 \
  --GT_path ../DIV2K/DIV2K_train_HR

# Quick test run (verify pipeline works, ~minutes)
python main.py --mode train \
  --pre_train_epoch 10 --fine_train_epoch 5 \
  --batch_size 4 --in_memory False \
  --LR_path ../DIV2K/DIV2K_train_LR_bicubic/X4 \
  --GT_path ../DIV2K/DIV2K_train_HR

# Resume fine-tuning from a saved checkpoint
python main.py --mode train --fine_tuning True \
  --generator_path ./model/pre_trained_model_0800.pt \
  --LR_path ../DIV2K/DIV2K_train_LR_bicubic/X4 \
  --GT_path ../DIV2K/DIV2K_train_HR
```

Checkpoints saved to `model/` every 800 (pretrain) / 500 (finetune) epochs.

### Training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `train` | `train`, `test`, or `test_only` |
| `--pre_train_epoch` | 8000 | L2 pretraining epochs |
| `--fine_train_epoch` | 4000 | GAN fine-tuning epochs |
| `--batch_size` | 16 | Reduce if out of GPU memory |
| `--patch_size` | 24 | LR crop size during training |
| `--scale` | 4 | Upscaling factor |
| `--num_workers` | 0 | Dataloader workers (increase on Linux) |
| `--in_memory` | True | Load full dataset into RAM |
| `--fine_tuning` | False | Resume from pretrained generator |
| `--generator_path` | — | Path to generator checkpoint |

---

## 4. TensorFlow SRGAN (`projects/tf-srgan-div2k/`)

> TFRecords and trained models are already included in the repo — you can run inference immediately after installing TensorFlow.

### Inference (models already included)

```bash
pip install tensorflow
cd projects/tf-srgan-div2k
python inference.py
```

Output images saved to `outputs/images/`.

### Training from scratch

```bash
pip install tensorflow tensorflow-datasets
cd projects/tf-srgan-div2k

# Step 1 — create TFRecords (one-time, auto-downloads DIV2K ~4GB)
python create_tfrecords.py

# Step 2 — train
python train_srgan.py
```

### Config settings (`src/config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `PRETRAIN_EPOCHS` | 2500 | Generator-only pretraining |
| `FINETUNE_EPOCHS` | 2500 | GAN fine-tuning |
| `TRAIN_BATCH_SIZE` | 64 | Reduce to 16 if out of GPU memory |
| `STEPS_PER_EPOCH` | 10 | Steps per epoch |
| `DIV2K_PATH` | `~/tfds` | Dataset download location |

For a quick smoke-test, set `PRETRAIN_EPOCHS=5`, `FINETUNE_EPOCHS=5`, `STEPS_PER_EPOCH=2` in `config.py`.

---

## Folder Structure

```
SuperResolution-Master/
├── HOW_TO_RUN.md                  ← you are here
├── requirements.txt               ← pip install -r requirements.txt
├── .gitignore
├── apps/
│   └── streamlit-esrgan-demo/
│       ├── app.py                 ← streamlit run app.py
│       ├── test.py                ← inference (called by app.py)
│       ├── RRDBNet_arch.py        ← ESRGAN architecture
│       └── weights/               ← model auto-downloaded here (git-ignored)
└── projects/
    ├── torch-srgan/
    │   ├── main.py                ← python main.py --mode train/test/test_only
    │   ├── mode.py                ← training/testing logic
    │   ├── pretrained_models/     ← SRGAN.pt + SRResNet.pt (included)
    │   ├── model/                 ← checkpoints saved here (git-ignored)
    │   └── result/                ← inference output (git-ignored)
    └── tf-srgan-div2k/
        ├── create_tfrecords.py    ← run once to prep data
        ├── train_srgan.py         ← main training script
        ├── inference.py           ← visualise results
        ├── outputs/               ← trained models + sample images (included)
        └── src/config.py          ← all hyperparameters and paths
```
