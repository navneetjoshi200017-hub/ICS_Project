# CIFAR-10 Image Classification Project
### Using MobileNetV1 Architecture | TensorFlow / Keras
*Academic Deep Learning Project*

---

## 1. Project Overview

This project implements an image classification system that can identify objects in images across 10 different categories. It was built entirely from scratch using Python and TensorFlow/Keras.

The system loads and preprocesses the CIFAR-10 dataset, trains a MobileNetV1 model with a custom loss function, and evaluates performance on unseen test images.

---

## 2. Model

| Property | Value |
|---|---|
| **Model Name** | MobileNetV1 (built from scratch — no pretrained weights) |
| **Framework** | TensorFlow 2.x / Keras |
| **Input Size** | 32 x 32 x 3 (RGB images) |
| **Output Classes** | 10 |
| **Width Multiplier (Alpha)** | 0.75 |

MobileNetV1 is a lightweight deep learning architecture designed for computational efficiency. Instead of standard convolutions, it uses Depthwise Separable Convolutions which split the operation into two cheaper steps:

1. **Depthwise Convolution** — filters each input channel independently
2. **Pointwise Convolution** — mixes channels together using a 1x1 convolution

This approach is approximately 8–9x computationally cheaper than a standard CNN while achieving comparable accuracy, making it well-suited for smaller datasets like CIFAR-10.

---

## 3. Dataset

| Property              | Value                                     |
|-----------------------|-------------------------------------------|
| **Dataset Name**      | CIFAR-10                                  |
| **Source**            | tf.keras.datasets.cifar10                 |
| **Total Images**      | 60,000 (50,000 training + 10,000 testing) |
| **Image Size**        | 32 x 32 pixels (RGB)                      |
| **Number of Classes** | 10                                        |

### Classes

| Index | Class      | Index | Class |
|-------|------------|-------|-------|
| 0     | Airplane   | 5     | Dog   |
| 1     | Automobile | 6     | Frog  |
| 2     | Bird       | 7     | Horse |
| 3     | Cat        | 8     | Ship  |
| 4     | Deer       | 9     | Truck |

### Data Split

| Split      | Size                        | Augmentation Applied |
|------------|-----------------------------|----------------------|
| Training   | 45,000 images               | Yes                  |
| Validation | 5,000 images (10% of train) | No                   |
| Testing    | 10,000 images               | No                   |

---

## 4. Loss Function

**Name:** ConfidenceCapLoss (Custom)

This is a custom loss function built on top of the standard Sparse Categorical Cross-Entropy. It works in two parts:

**Part 1 — Base Loss (Sparse Categorical Cross-Entropy)**
The standard classification loss. For each image it looks at the probability assigned to the correct class. A high probability gives a small loss; a low probability gives a large loss. This is what teaches the model to predict the correct class.

**Part 2 — Confidence Cap Penalty**
An extra penalty that discourages the model from being overconfident. A confidence cap of 0.85 is set. If the model's top prediction exceeds 85%, the excess amount is penalised.

| Scenario  | Model Confidence | Excess                  | Penalty Applied |
|-----------|------------------|-------------------------|-----------------|
| Above cap | 0.95 (95%)       | 0.95 - 0.85 = 0.10      | Yes             |
| Under cap | 0.70 (70%)       | max(0, 0.70 - 0.85) = 0 | No              |

```
Final Loss = Cross-Entropy + ( 0.1 x Confidence Cap Penalty )
```

**Key Parameters (adjustable in config.py):**
- `penalty_weight = 0.1` — controls how strongly the penalty is applied
- `confidence_cap = 0.85` — the confidence threshold above which penalty kicks in

**Why this helps:**
Overconfident models tend to generalise poorly on unseen data. By capping confidence, the model stays slightly humble, which improves real-world performance on test images.

---

## 5. Results

| Metric                      | Value                                           |
|-----------------------------|-------------------------------------------------|
| **Final Test Accuracy**     | ~83–86%                                         |
| **Training Epochs**         | 25                                              |
| **Batch Size**              | 64                                              |
| **Optimizer**               | Adam (learning rate = 0.001)                    |
| **Data Augmentation**       | Enabled (flip, crop, brightness, contrast, hue) |
| **Early Stopping Patience** | 5 epochs                                        |

---

## 6. Project Structure

| File / Folder               | Purpose                                                          |
|-----------------------------|------------------------------------------------------------------|
| `config.py`                 | All hyperparameters in one place                                 |
| `train.py`                  | Main training script — run this to train                         |
| `inference.py`              | Load saved model and run predictions on test images              |
| `models/mobilenet_v1.py`    | MobileNetV1 architecture built from scratch                      |
| `models/loss.py`            | Custom ConfidenceCapLoss function                                |
| `utils/data_loader.py`      | CIFAR-10 loading, preprocessing, augmentation pipeline           |
| `utils/plotting.py`         | Matplotlib accuracy and loss curve plots                         |
| `outputs/best_model.keras`  | Saved best model weights (auto-generated after training)         |
| `outputs/accuracy_plot.png` | Training vs validation accuracy graph                            |
| `outputs/loss_plot.png`     | Training vs validation loss graph                                |
| `inferenceoutput/`          | Screenshots of output of inference.py for different test samples |

---

## 7. Team Members and Contributions

### K Vignesh — Model Architecture & Training Pipeline
- Built the MobileNetV1 architecture from scratch (mobilenet_v1.py)
- Implemented depthwise separable convolution blocks
- Set up the full training pipeline in train.py
- Configured model callbacks — early stopping, checkpoint, learning rate scheduler
- Ran and monitored all training experiments

### Shubham Kumar — Data Pipeline & Visualisation
- Implemented CIFAR-10 data loading and preprocessing (data_loader.py)
- Built the data augmentation pipeline (flip, crop, brightness, contrast, hue)
- Created the tf.data training, validation, and test pipelines
- Wrote the Matplotlib plotting functions (plotting.py)
- Generated and analysed the accuracy and loss training graphs

### Navneet Joshi — Custom Loss Function & Evaluation
- Researched and designed the custom ConfidenceCapLoss function (loss.py)
- Implemented and tested multiple loss function variants
- Wrote the inference script for test image predictions (inference.py)
- Evaluated the final model on the test set and recorded results
- Documented the project (README, code comments)

---

## 8. Dependencies

| Package    | Minimum Version | Purpose                                 |
|------------|-----------------|-----------------------------------------|
| tensorflow | 2.12.0          | Model building, training, and inference |
| matplotlib | 3.7.0           | Plotting accuracy and loss curves       |
| numpy      | 1.23.0          | Array operations and data handling      |

---

## 9. Notes

- The CIFAR-10 dataset is automatically downloaded on first run (~170 MB)
- The best model checkpoint is saved automatically to `outputs/best_model.keras`
- Plots are saved as PNG files to the `outputs/` folder after training completes
- `inference.py` takes random 10 testing images using `random.sample()` so that the output is different each time the code is run
