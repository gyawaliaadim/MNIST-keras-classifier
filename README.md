# 🚀 MNIST MLP Classifier

A baseline handwritten digit classification project using a Multilayer Perceptron (MLP) built with TensorFlow and Keras.
The model is trained and evaluated on the classic MNIST dataset, demonstrating the end-to-end deep learning workflow from preprocessing to evaluation.

This repository intentionally keeps the architecture simple to serve as a clear, interpretable starting point before moving to more advanced models like CNNs.

## 🗂️ Project Overview

The goal of this project is to classify grayscale images of handwritten digits (0–9).
Each image is 28×28 pixels and is flattened before being passed through a fully connected neural network.

## 🎯 This project focuses on:
- Clean data preprocessing
- Correct loss–label alignment
- A minimal yet effective neural network architecture
- Reproducible training and evaluation

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- MNIST Dataset
-
## ⚙️ Model Architecture
- Input Layer: 28×28 images, flattened to 784 features
- Hidden Layer: Dense layer with ReLU activation
- Output Layer: Dense layer with Softmax activation (10 classes)

This architecture represents a classic MLP baseline for image classification.

## ℹ️ Training Details
- Optimizer: Adam
- Loss Function: Categorical Cross-Entropy
- Batch Size: 12
- Epochs: 5

Evaluation Metric: Accuracy
Pixel values are normalized to the range [0, 1], and labels are one-hot encoded.

## 📝 Results
- The trained model achieves ~96–97% accuracy on the MNIST test set, which is expected for a dense neural network without convolutional layers.
- This performance validates the pipeline while leaving room for architectural improvements.

## 🚀 How to Run

### Clone the repository
```bash
git clone [https://github.com/your-username/mnist-mlp-classifier.git](https://github.com/gyawaliaadim/MNIST-keras-classifier)
cd MNIST-keras-classifier
```

### Install dependencies
```bash
pip install tensorflow numpy
```

### Run the training script
```bash
python main.py
```

## -By Aadim Gyawali
