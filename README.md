

# 📍 DRIR-Net: Dual-Branch Rotation Invariant and Robust Network for 3D Place Recognition

> A robust and rotation-equivariant approach for place recognition in large-scale 3D environments.  

---

## 🌟 Highlights

- ✅ A subregion downsampling method reduces preprocessing time for point clouds
- ✅ The extraction of noise-resistant point features and rotation-invariant descriptors
- ✅ Subregion cross attention fuses point and voxel features with accurate correspondence
- ✅ Superior performance and generalization on multiple 3D place recognition datasets

---

## 🏗️ Architecture Overview

![Architecture](figures/pipline.jpg)

---

## 🔧 Setup

### Environment

This project was tested using the following environment:

- **OS**: Ubuntu 20.04  
- **Python**: 3.8  
- **CUDA**: 11.7  
- **PyTorch**: 1.7  
- **MinkowskiEngine**: 0.5.0  

### Dependencies

Install the required Python packages:

```bash
pip install torch==1.7.0
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine@v0.5.0
pip install pytorch_metric_learning>=0.9.94
pip install numba
pip install tensorboard
pip install pandas
pip install psutil
pip install bitarray

```
---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/IATBOMSW/DRIR-Net.git
cd DRIR-Net
```

### Dataset preparation

```bash
cd generating_queries/

# Generate training tuples for the USyd Dataset
python generate_training_tuples_usyd.py

# Generate evaluation tuples for the USyd Dataset
python generate_test_sets_usyd.py
```

### Training

```bash
cd training/

# To train the desired model on the USyd Dataset
python train.py --config ../config/config_usyd.txt --model_config ../config/model_config_usyd.txt
```

### Evaluation

```bash
cd eval/

python evaluate.py --config ../config/config_usyd.txt --model_config ../config/model_config_usyd.txt --weights ../weights/weights.pth
```

---


## 📄 Citation

This section is under construction.

---

## 🙏 Acknowledgements

We are thankful for the remarkable work of [MinkLoc3D-SI](https://github.com/KamilZywanowski/MinkLoc3D-SI), [CASSPR](https://github.com/Yan-Xia/CASSPR), and the  community, which inspired this research.

