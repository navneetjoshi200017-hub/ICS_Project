import matplotlib.pyplot as plt
from config import PLOT_ACC_PATH, PLOT_LOSS_PATH


def plot_accuracy(history, save: bool = True):
    """
    Plots epoch-wise training accuracy and validation accuracy.

    Args:
        history : History object returned by model.fit()
        save    : If True, saves the figure to PLOT_ACC_PATH
    """
    train_acc = history.history["accuracy"]
    val_acc   = history.history["val_accuracy"]
    epochs    = range(1, len(train_acc) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_acc, "b-o", linewidth=1.8, markersize=4, label="Training Accuracy")
    plt.plot(epochs, val_acc,   "r-s", linewidth=1.8, markersize=4, label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save:
        plt.savefig(PLOT_ACC_PATH, dpi=150)
        print(f"Accuracy plot saved → {PLOT_ACC_PATH}")

    plt.show()


def plot_loss(history, save: bool = True):
    """
    Plots epoch-wise training loss and validation loss.

    Args:
        history : History object returned by model.fit()
        save    : If True, saves the figure to PLOT_LOSS_PATH
    """
    train_loss = history.history["loss"]
    val_loss   = history.history["val_loss"]
    epochs     = range(1, len(train_loss) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_loss, "b-o", linewidth=1.8, markersize=4, label="Training Loss")
    plt.plot(epochs, val_loss,   "r-s", linewidth=1.8, markersize=4, label="Validation Loss")
    plt.title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save:
        plt.savefig(PLOT_LOSS_PATH, dpi=150)
        print(f"Loss plot saved → {PLOT_LOSS_PATH}")

    plt.show()


def plot_inference(images, true_labels, pred_labels, class_names, n: int = 10):
    """
    Shows a grid of test images with true and predicted labels.

    Args:
        images      : numpy array of shape (N, 32, 32, 3), values in [0, 1]
        true_labels : integer array of shape (N,)
        pred_labels : integer array of shape (N,)
        class_names : list of class name strings
        n           : number of images to display (max 10)
    """
    n = min(n, len(images))
    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(images[i])
        colour = "green" if true_labels[i] == pred_labels[i] else "red"
        axes[i].set_title(
            f"True : {class_names[true_labels[i]]}\n"
            f"Pred : {class_names[pred_labels[i]]}",
            fontsize=8, color=colour
        )
        axes[i].axis("off")

    plt.suptitle("Sample Inference Results (green = correct, red = wrong)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()
