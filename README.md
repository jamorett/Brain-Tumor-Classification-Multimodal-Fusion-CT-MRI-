# 🧠 Multimodal Brain Tumor Classification: CT & MRI Fusion

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.X-FF6F00?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri)

## 📖 Summary

This repository implements a **multimodal deep learning pipeline** for the binary classification of brain tumors ("Healthy" vs. "Tumor"). The core innovation of this project is the implementation of **Input-Level Data Fusion**, combining spatially aligned Computerized Tomography (CT) and Magnetic Resonance Imaging (MRI) scans into a single tensor representation.

The system performs a comparative analysis between two architectural approaches:
1.  **Transfer Learning:** Leveraging a pre-trained **VGG16** backbone.
2.  **Custom Architecture:** A lightweight, bespoke **Convolutional Neural Network (CNN)** trained from scratch.

## 🏗️ Technical Architecture

### 1. Data Fusion Strategy (Input-Level)
Unlike late fusion (voting) or feature-level fusion, this pipeline implements **pixel-level fusion** during the preprocessing stage.
* **Input:** Paired MRI ($I_{mri}$) and CT ($I_{ct}$) images.
* **Operation:** Pixel-wise averaging after normalization.
    $$I_{fused} = \frac{I_{mri} + I_{ct}}{2.0}$$
* **Rationale:** CT provides superior bone/calcification detail, while MRI excels at soft tissue contrast. Combining them aims to provide the model with a richer feature set than either modality alone.

### 2. Model Architectures

| Feature | Model A: VGG16 (Transfer Learning) | Model B: Custom CNN |
| :--- | :--- | :--- |
| **Backbone** | VGG16 (Weights: ImageNet) | 3x Conv2D Blocks (32, 64, 128 filters) |
| **Trainable Params** | ~131,000 (Frozen Backbone) | ~1.6 Million (Fully Trainable) |
| **Regularization** | Dropout (0.3) | Dropout (0.5), MaxPolling |
| **Input Shape** | (224, 224, 3) | (224, 224, 3) |
| **Optimizer** | Adam ($\alpha = 1e-3$) | Adam ($\alpha = 1e-3$) |

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.8+
* CUDA-enabled GPU (Recommended for training)

### Dependencies
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn kagglehub
```

### Dataset

The script utilizes the kagglehub library to automatically download the Brain Tumor Multimodal Image Dataset. No manual download is required; the script handles caching.

## 🚀 Usage

Execute the main training script:

```bash
python brain_tumor_fusion.py
```

### Pipeline Flow:

  - Ingestion: Automatic dataset download and path validation.

  - Pairing: Algorithmic alignment of MRI and CT images based on filenames.

  - Generator Initialization: FusedDataGenerator creates batches of fused images in real-time.

  - Training: Sequential training of VGG16 followed by the Custom CNN.

  - Evaluation: Generation of Confusion Matrices and Classification Reports.


## 📊 Evaluation Metrics

The system outputs the following performance indicators for both models:

  - Binary Cross-Entropy Loss

  - Accuracy Score

  - Confusion Matrix: To visualize False Positives (Type I Error) vs. False Negatives (Type II Error).

  - Training Dynamics: Plots of Loss/Accuracy over epochs to identify overfitting.
