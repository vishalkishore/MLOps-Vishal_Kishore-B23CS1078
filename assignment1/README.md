# Assignment 1: Deep Learning Model Training on MNIST and FashionMNIST

[![Colab](https://img.shields.io/badge/Open%20in-Colab-orange)](https://colab.research.google.com/drive/11LnNZgeAUmjPcV2wpFzFp2L1hn3mjEKd?usp=sharing)
[![GitHub Pages](https://img.shields.io/badge/GitHub-Pages-blue)](https://vishalkishore.github.io/MLOps-Vishal_Kishore-B23CS1078/)


## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 📓 **Colab Notebook** | [Open in Colab](https://colab.research.google.com/drive/11LnNZgeAUmjPcV2wpFzFp2L1hn3mjEKd?usp=sharing) |
| 📄 **Report** | [View PDF](../../blob/Assignment-1/report/B23CS1078_Vishal_Kishore_Ass1.pdf) |
| 💾 **Best Model** | [Download](../../blob/Assignment-1/models/best_model.pth) |
| 📊 **Results CSV** | [View](../../blob/Assignment-1/results/results_summary.csv) |

---

## 📋 Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Datasets** | MNIST, FashionMNIST |
| **Models** | ResNet-18, ResNet-34, ResNet-50 (pretrained=False) |
| **Data Split** | 70%-10%-20% (Train-Val-Test) |
| **Mixed Precision** | USE_AMP = True |
| **Early Stopping** | 5 epochs |
| **Seeds** | 35, 78, 13 |

---

## 📊 Q1(a): Classification Results

### MNIST Dataset - 20 Epochs

| Batch Size | Optimizer | LR | ResNet-18 (No Sched.) | ResNet-18 (Cosine) | ResNet-50 (Cosine) |
|:----------:|:---------:|:----:|:---------------------:|:------------------:|:------------------:|
| 16 | SGD | 0.001 | 99.20 ± 0.10% | 99.32 ± 0.04% | 99.07 ± 0.07% |
| 16 | SGD | 0.0001 | 98.96 ± 0.09% | 98.90 ± 0.18% | 98.50 ± 0.04% |
| 16 | Adam | 0.001 | 99.23 ± 0.04% | **99.40 ± 0.05%** | 99.26 ± 0.09% |
| 16 | Adam | 0.0001 | 99.12 ± 0.14% | 99.22 ± 0.19% | 99.19 ± 0.10% |
| 32 | SGD | 0.001 | 98.63 ± 0.10% | 99.18 ± 0.09% | 99.01 ± 0.03% |
| 32 | SGD | 0.0001 | 99.12 ± 0.11% | 98.57 ± 0.11% | 97.99 ± 0.08% |
| 32 | Adam | 0.001 | 99.30 ± 0.11% | **99.45 ± 0.07%** | 99.27 ± 0.11% |
| 32 | Adam | 0.0001 | 99.12 ± 0.14% | 99.22 ± 0.19% | 98.96 ± 0.10% |

### MNIST Dataset - 10 Epochs

| Batch Size | Optimizer | LR | ResNet-18 (No Sched.) | ResNet-18 (Cosine) | ResNet-50 (No Sched.) | ResNet-50 (Cosine) |
|:----------:|:---------:|:----:|:---------------------:|:------------------:|:---------------------:|:------------------:|
| 16 | SGD | 0.001 | 99.14 ± 0.05% | 99.20 ± 0.06% | 98.81 ± 0.10% | 98.99 ± 0.05% |
| 16 | SGD | 0.0001 | 98.84 ± 0.04% | 98.76 ± 0.06% | 98.32 ± 0.10% | 98.12 ± 0.11% |
| 16 | Adam | 0.001 | 99.23 ± 0.04% | **99.45 ± 0.06%** | 98.90 ± 0.10% | 99.30 ± 0.10% |
| 16 | Adam | 0.0001 | 99.10 ± 0.04% | 99.30 ± 0.07% | 98.53 ± 0.09% | 98.89 ± 0.15% |
| 32 | SGD | 0.001 | 99.13 ± 0.09% | 99.14 ± 0.10% | 98.74 ± 0.11% | 98.92 ± 0.03% |
| 32 | SGD | 0.0001 | 98.55 ± 0.13% | 98.44 ± 0.04% | 97.68 ± 0.09% | 97.48 ± 0.11% |
| 32 | Adam | 0.001 | 99.09 ± 0.04% | **99.41 ± 0.03%** | 98.77 ± 0.05% | 99.27 ± 0.11% |
| 32 | Adam | 0.0001 | 99.01 ± 0.09% | 99.25 ± 0.19% | 98.27 ± 0.12% | 98.64 ± 0.05% |

### FashionMNIST Dataset - 20 Epochs

| Batch Size | Optimizer | LR | ResNet-18 (Cosine) | ResNet-50 (Cosine) |
|:----------:|:---------:|:----:|:------------------:|:------------------:|
| 16 | SGD | 0.001 | 91.73 ± 0.06% | 90.40 ± 0.50% |
| 16 | SGD | 0.0001 | 90.12 ± 0.37% | 88.66 ± 0.36% |
| 16 | Adam | 0.001 | **92.04 ± 0.07%** | **91.90 ± 0.11%** |
| 16 | Adam | 0.0001 | 91.42 ± 0.09% | 88.66 ± 0.36% |
| 32 | SGD | 0.001 | 91.28 ± 0.22% | 89.73 ± 0.16% |
| 32 | SGD | 0.0001 | 89.44 ± 0.04% | 87.40 ± 0.36% |
| 32 | Adam | 0.001 | **92.12 ± 0.09%** | 91.69 ± 0.26% |
| 32 | Adam | 0.0001 | 91.18 ± 0.21% | 89.87 ± 0.19% |

### FashionMNIST Dataset - 10 Epochs

| Batch Size | Optimizer | LR | ResNet-18 (No Sched.) | ResNet-18 (Cosine) | ResNet-50 (No Sched.) | ResNet-50 (Cosine) |
|:----------:|:---------:|:----:|:---------------------:|:------------------:|:---------------------:|:------------------:|
| 16 | SGD | 0.001 | 90.89 ± 0.29% | 91.61 ± 0.14% | 89.50 ± 0.31% | 90.04 ± 0.64% |
| 16 | SGD | 0.0001 | 89.70 ± 0.04% | 88.71 ± 0.15% | 87.04 ± 0.23% | 86.70 ± 0.49% |
| 16 | Adam | 0.001 | 91.34 ± 0.13% | **92.28 ± 0.09%** | 89.24 ± 1.59% | 91.90 ± 0.15% |
| 16 | Adam | 0.0001 | 90.80 ± 0.27% | 91.24 ± 0.13% | 87.82 ± 0.49% | 88.37 ± 0.69% |
| 32 | SGD | 0.001 | 90.47 ± 0.22% | 91.05 ± 0.36% | 88.48 ± 0.53% | 89.02 ± 0.36% |
| 32 | SGD | 0.0001 | 88.93 ± 0.29% | 88.80 ± 0.37% | 85.75 ± 0.16% | 85.41 ± 0.30% |
| 32 | Adam | 0.001 | 91.03 ± 0.13% | **92.22 ± 0.19%** | 88.81 ± 0.46% | 91.47 ± 0.37% |
| 32 | Adam | 0.0001 | 90.19 ± 0.20% | 90.67 ± 0.40% | 87.32 ± 0.37% | 88.65 ± 0.46% |

---

## 📊 Q1(b): SVM Classifier Results

### MNIST - RBF Kernel

| C | Gamma | Degree | Accuracy (%) | Training Time (ms) |
|:-:|:-----:|:------:|:------------:|:------------------:|
| 1 | scale | 3 | 94.45 | 7993.38 |
| 5 | scale | 3 | **95.40** | 7411.84 |
| 10 | scale | 3 | **95.55** | 8001.53 |
| 1 | 0.01 | 3 | 93.25 | 7175.44 |
| 5 | 0.01 | 3 | 95.10 | 9403.81 |
| 10 | 0.01 | 3 | **95.45** | 6162.32 |

### MNIST - Polynomial Kernel

| C | Gamma | Degree | Accuracy (%) | Training Time (ms) |
|:-:|:-----:|:------:|:------------:|:------------------:|
| 1 | scale | 2 | 94.15 | 6056.24 |
| 5 | scale | 2 | 95.05 | 5633.33 |
| 10 | scale | 2 | **95.20** | 5204.85 |
| 1 | scale | 3 | 93.40 | 8414.45 |
| 5 | scale | 3 | 94.00 | 6237.90 |
| 10 | scale | 3 | 94.10 | 5884.44 |

### FashionMNIST - RBF Kernel

| C | Gamma | Degree | Accuracy (%) | Training Time (ms) |
|:-:|:-----:|:------:|:------------:|:------------------:|
| 1 | scale | 3 | 86.00 | 10334.46 |
| 5 | scale | 3 | **87.45** | 6820.15 |
| 10 | scale | 3 | 87.35 | 6757.70 |
| 1 | 0.01 | 3 | 86.00 | 7202.21 |
| 5 | 0.01 | 3 | **87.45** | 6617.26 |
| 10 | 0.01 | 3 | 87.35 | 6687.58 |

### FashionMNIST - Polynomial Kernel

| C | Gamma | Degree | Accuracy (%) | Training Time (ms) |
|:-:|:-----:|:------:|:------------:|:------------------:|
| 1 | scale | 2 | 84.75 | 6769.21 |
| 5 | scale | 2 | 86.60 | 5886.71 |
| 10 | scale | 2 | **86.80** | 6729.06 |
| 1 | scale | 3 | 82.80 | 8146.13 |
| 5 | scale | 3 | 85.35 | 6939.22 |
| 10 | scale | 3 | 85.75 | 6051.58 |

---

## 📊 Q2: CPU vs GPU Performance Comparison

### FashionMNIST Dataset - CPU vs GPU Analysis

| Compute | Batch Size | Optimizer | LR | R-18 Acc | R-34 Acc | R-50 Acc | R-18 Time (ms) | R-34 Time (ms) | R-50 Time (ms) |
|:-------:|:----------:|:---------:|:----:|:--------:|:--------:|:--------:|:--------------:|:--------------:|:--------------:|
| CPU | 16 | SGD | 0.001 | 91.74 ± 0.06% | 91.65 ± 0.38% | 90.39 ± 0.50% | 7,274,210 | 1,879,214 | 2,619,201 |
| CPU | 16 | Adam | 0.001 | 92.01 ± 0.06% | 92.08 ± 0.25% | 91.90 ± 0.10% | 7,679,113 | 1,867,889 | 2,699,217 |
| GPU | 16 | SGD | 0.001 | 91.73 ± 0.06% | 91.65 ± 0.38% | 90.40 ± 0.50% | 545,392 | 923,336 | 1,663,622 |
| GPU | 16 | Adam | 0.001 | **92.04 ± 0.07%** | **92.08 ± 0.25%** | 91.90 ± 0.11% | 533,329 | 1,025,271 | 1,663,071 |

### FLOPs Analysis

| Compute Device | Batch Size | ResNet-18 FLOPs | ResNet-34 FLOPs | ResNet-50 FLOPs |
|:--------------:|:----------:|:---------------:|:---------------:|:---------------:|
| CPU | 16 | 0.03G | 0.07G | 0.08G |
| GPU | 16 | 0.03G | 0.07G | 0.08G |

---

## 🏆 Best Model Summary

| Dataset | Best Model | Configuration | Test Accuracy |
|:-------:|:----------:|:-------------:|:-------------:|
| MNIST | ResNet-18 | Adam, LR=0.001, Batch=32, Cosine, 20 Epochs | **99.45 ± 0.07%** |
| FashionMNIST | ResNet-18 | Adam, LR=0.001, Batch=16, Cosine, 10 Epochs | **92.28 ± 0.09%** |

---

## 👤 Author

**Vishal Kishore** | Roll Number: B23CS1078
