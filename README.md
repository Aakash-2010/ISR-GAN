# SuperResolution — GAN-based Image Upscaling

Implementation of SRGAN and ESRGAN from scratch using PyTorch and TensorFlow.

---

## Demo

<!-- Add a screenshot or GIF of the Streamlit app here -->
<!-- Example: ![Demo](docs/images/demo.gif) -->

## Results

| Low Resolution | Super Resolution (4x) |
|---|---|
| <!-- ![LR sample](docs/images/lr_sample.png) --> | <!-- ![SR sample](docs/images/sr_sample.png) --> |

<!-- Add more before/after comparisons below -->

---

## Project Structure

```
SuperResolution/
├── apps/
│   └── streamlit-esrgan-demo/   # Interactive web demo (ESRGAN)
├── projects/
│   ├── torch-srgan/             # PyTorch SRGAN implementation
│   └── tf-srgan-div2k/          # TensorFlow SRGAN implementation
└── requirements.txt
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the web demo

```bash
cd apps/streamlit-esrgan-demo
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload any image and download the 4x upscaled result.

### 3. Train / run inference (PyTorch)

```bash
cd projects/torch-srgan

# Inference only (no GPU needed)
python main.py --mode test_only \
    --generator_path ./pretrained_models/SRGAN.pt \
    --LR_path /path/to/images

# Full training
python main.py --mode train
```

### 4. Train / run inference (TensorFlow)

```bash
cd projects/tf-srgan-div2k

python train_srgan.py   # train
python inference.py     # run inference
```

## Architecture

- **SRGAN** — residual network generator + PatchGAN discriminator with perceptual loss (VGG feature matching)
- **ESRGAN / RealESRGAN** — enhanced SRGAN with RRDB blocks, served via the Streamlit demo using a pretrained HuggingFace model

## References

- [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
- [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
