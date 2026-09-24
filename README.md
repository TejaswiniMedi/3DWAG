# 3D-WAG

**3D-WAG: Hierarchical Wavelet-Guided Autoregressive Generation for High-Fidelity 3D Shapes**

**BMVC 2025**

**Authors:**
Tejaswini Medi, Arianna Rampini, Pradyumna Reddy,
Pradeep Kumar Jayaraman, Margret Keuper

3D-WAG is an autoregressive framework for high-fidelity 3D shape generation using hierarchical multi-scale wavelet representations and next-scale token-map prediction.

The framework supports:

* Unconditional 3D generation
* Class-conditional 3D generation
* Text-conditioned 3D generation

---

## Installation

Clone the repository and create the environment:

```bash
git clone https://github.com/TejaswiniMedi/3DWAG.git
cd 3DWAG

conda create -n 3dwag python=3.10
conda activate 3dwag
```

---

## Training

### Train the VQ-VAE

```bash
python train_vqvae.py \
    --config configs/vqvae.yaml
```

### Train the Autoregressive Model

After training the VQ-VAE, train the autoregressive model using the VQ-VAE checkpoint:

```bash
python train_ar.py \
    --vqvae_ckpt checkpoints/vqvae.pt
```

---

## Generation

### Unconditional Generation

Generate 3D shapes using the trained VQ-VAE and autoregressive model:

```bash
python generate.py \
    --vqvae_ckpt checkpoints/vqvae.pt \
    --ar_ckpt checkpoints/ar.pt \
    --output_dir outputs/
```

### Class-Conditional Generation

For class-conditioned generation, specify the target class:

```bash
python generate.py \
    --vqvae_ckpt checkpoints/vqvae.pt \
    --ar_ckpt checkpoints/ar.pt \
    --class_name chair \
    --output_dir outputs/chair/
```

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{Medi_2025_BMVC,
  author    = {Tejaswini Medi and Arianna Rampini and Pradyumna Reddy and
               Pradeep Kumar Jayaraman and Margret Keuper},
  title     = {3D-WAG: Hierarchical Wavelet-Guided Autoregressive Generation
               for High-Fidelity 3D Shapes},
  booktitle = {36th British Machine Vision Conference 2025, BMVC 2025},
  publisher = {BMVA},
  year      = {2025}
}
```

---

## Acknowledgements

This repository contains the implementation of **3D-WAG**, presented at the **British Machine Vision Conference (BMVC) 2025**.


