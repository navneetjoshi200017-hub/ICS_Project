# Edit values here to change hyperparameters(Global parameters) across the entire project.

# Dataset 
NUM_CLASSES     = 10
INPUT_SHAPE     = (32, 32, 3)   # CIFAR-10 native resolution

# MobileNetV1 
ALPHA           = 0.75          # Width multiplier: 1.0 | 0.75 | 0.5

# Training 
BATCH_SIZE      = 64
EPOCHS          = 25            # Fewer epochs → stops before squeezing the last 3-4%
LEARNING_RATE   = 1e-3


# Paths 
CHECKPOINT_PATH = "outputs/best_model.keras"
PLOT_ACC_PATH   = "outputs/accuracy_plot.png"
PLOT_LOSS_PATH  = "outputs/loss_plot.png"

#  Augmentation 
USE_AUGMENTATION = True
