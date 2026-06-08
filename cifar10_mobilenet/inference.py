# inference.py — Load the saved model and run inference on test images
import os
import sys
import numpy as np
import tensorflow as tf
import random

sys.path.insert(0, os.path.dirname(__file__))

from config import CHECKPOINT_PATH
from models.loss import ConfidenceCapLoss
from utils.data_loader import load_cifar10, CIFAR10_CLASSES
from utils.plotting import plot_inference


_, (x_test, y_test) = load_cifar10()    #Load CIFAR-10 test data


x_test_norm = x_test.astype("float32") / 255.0  # Normalize pixel values to [0, 1]

print(f"Loading model from: {CHECKPOINT_PATH}")

# Load saved model
model = tf.keras.models.load_model(
    CHECKPOINT_PATH,
    custom_objects={"ConfidenceCapLoss": ConfidenceCapLoss}
)

# Run inference on 10 random test images
start = random.randint(0, len(x_test_norm) - 10)
sample_images = x_test_norm[start:start+10]
sample_labels = y_test[start:start+10]

predictions   = model.predict(sample_images, verbose=0)
pred_labels   = np.argmax(predictions, axis=1)

# Print results
print("\nInference results:")
print(f"{'Index':<6} {'True Label':<15} {'Pred Label':<15} {'Confidence':>10}")
print("-" * 50)
for i in range(10):
    true_name = CIFAR10_CLASSES[sample_labels[i]]
    pred_name = CIFAR10_CLASSES[pred_labels[i]]
    confidence = predictions[i, pred_labels[i]] * 100
    match = "✓" if sample_labels[i] == pred_labels[i] else "✗"
    print(f"{i:<6} {true_name:<15} {pred_name:<15} {confidence:>9.1f}%  {match}")

# Plot sample images with results
plot_inference(
    images=sample_images,
    true_labels=sample_labels,
    pred_labels=pred_labels,
    class_names=CIFAR10_CLASSES,
    n=10
)
