# train.py — Main training script for CIFAR-10 MobileNetV1

import os
import sys
import tensorflow as tf

# Make sure sub-packages are importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    EPOCHS, LEARNING_RATE, CHECKPOINT_PATH, ALPHA, NUM_CLASSES, INPUT_SHAPE
)
from utils.data_loader import load_cifar10, build_datasets
from utils.plotting    import plot_accuracy, plot_loss
from models.mobilenet_v1 import build_mobilenet_v1
from models.loss import ConfidenceCapLoss

# 1. Load & prepare data

print("=" * 60)
print("Loading CIFAR-10 dataset ...")
(x_train, y_train), (x_test, y_test) = load_cifar10()
print(f"  Train: {x_train.shape}, Test: {x_test.shape}")

train_ds, val_ds, test_ds = build_datasets(x_train, y_train, x_test, y_test)


# 2. Build model

print(f"\nBuilding MobileNetV1 (alpha={ALPHA}) ...")
model = build_mobilenet_v1(
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES,
    alpha=ALPHA
)
model.summary(line_length=80)


# 3. Compile

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fn   = ConfidenceCapLoss(penalty_weight=0.1, confidence_cap=0.85)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=["accuracy"]
)

# 4. Callbacks

os.makedirs("outputs", exist_ok=True)

callbacks = [
    # Save the best model (lowest validation loss)
    tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    # Stop early if val_loss stops improving for 5 epochs
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduce LR when val_loss plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),

    # Print a summary of each epoch to console
    tf.keras.callbacks.TerminateOnNaN(),
]


# 5. Train

print("\nStarting training ...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)


# 6. Evaluate on test set

print("\nEvaluating on test set ...")
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\n  Test Loss     : {test_loss:.4f}")
print(f"  Test Accuracy : {test_acc * 100:.2f}%")


# 7. Plot training curves

print("\nGenerating plots ...")
plot_accuracy(history, save=True)
plot_loss(history,    save=True)

print("\nDone! Plots saved to outputs/")
