# 3D-WAG


**3D-WAG: Hierarchical Wavelet-Guided Autoregressive Generation for High-Fidelity 3D Shapes**

**BMVC 2025**

Tejaswini Medi, Arianna Rampini, Pradyumna Reddy,  
Pradeep Kumar Jayaraman, Margret Keuper

3D-WAG is an autoregressive framework for high-fidelity 3D shape generation using
hierarchical multi-scale wavelet representations and next-scale token-map prediction.

The framework supports:

- Unconditional 3D generation
- Class-conditional 3D generation
- Text-conditioned 3D generation

---

# Installation

Clone the repository:

```bash
git clone https://github.com/TejaswiniMedi/3DWAG.git
cd 3DWAG

conda create -n 3dwag python=3.10
conda activate 3dwag


python train_vqvae.py --config configs/vqvae.yaml


python train_ar.py \
    --vqvae_ckpt checkpoints/vqvae.pt

python generate.py \
    --vqvae_ckpt checkpoints/vqvae.pt \
    --ar_ckpt checkpoints/ar.pt \
    --output_dir outputs/

python generate.py \
    --vqvae_ckpt checkpoints/vqvae.pt \
    --ar_ckpt checkpoints/ar.pt \
    --class_name chair \
    --output_dir outputs/chair/

```

## Citation
```bash
@inproceedings{Medi_2025_BMVC,
  author    = {Tejaswini Medi and Arianna Rampini and Pradyumna Reddy and
               Pradeep Kumar Jayaraman and Margret Keuper},
  title     = {3D-WAG: Hierarchical Wavelet-Guided Autoregressive Generation
               for High-Fidelity 3D Shapes},
  booktitle = {36th British Machine Vision Conference 2025, BMVC 2025},
  publisher = {BMVA},
  year      = {2025}
}
``
