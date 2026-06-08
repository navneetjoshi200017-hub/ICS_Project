
import tensorflow as tf
from config import NUM_CLASSES, INPUT_SHAPE, ALPHA


def _make_divisible(v: float, divisor: int = 8) -> int:
    """Round channel count to the nearest multiple of `divisor`."""
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


def _apply_alpha(channels: int, alpha: float) -> int:
    """Scale channels by the width multiplier (alpha)."""
    return _make_divisible(channels * alpha)



def conv_bn_relu(x, filters: int, kernel_size: int = 3,
                 strides: int = 1, padding: str = "same"):
    """
    Standard Convolution → BatchNorm → ReLU.
    Used only for the stem (first layer).
    """
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=strides,
        padding=padding, use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def depthwise_separable_block(x, pointwise_filters: int, strides: int = 1):
    """
    MobileNetV1 Depthwise Separable Convolution block.

    Structure:
        Depthwise Conv (3×3, groups=in_channels) → BN → ReLU
        Pointwise Conv (1×1)                     → BN → ReLU

    Args:
        x                 : Input tensor
        pointwise_filters : Number of output channels after 1×1 conv
        strides           : Stride for depthwise conv (1 or 2)
    """
    # Depthwise conv: each channel filtered independently
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3, strides=strides, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Pointwise conv: 1×1 conv to mix channels
    x = tf.keras.layers.Conv2D(
        pointwise_filters, kernel_size=1, strides=1,
        padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    return x


def build_mobilenet_v1(input_shape=INPUT_SHAPE,
                       num_classes=NUM_CLASSES,
                       alpha=ALPHA):
    """
    Builds a CIFAR-10-adapted MobileNetV1 model using the Keras Functional API.

    Args:
        input_shape : Tuple (H, W, C). Default (32, 32, 3).
        num_classes : Number of output classes. Default 10.
        alpha       : Width multiplier. Reduces channels by factor alpha.
                      Supported: 1.0 (full), 0.75, 0.5.

    Returns:
        tf.keras.Model
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # Stem: standard 3×3 conv — no downsampling (images are already small)
    x = conv_bn_relu(inputs, filters=_apply_alpha(32, alpha), kernel_size=3, strides=1)

    # MobileNetV1 body — depthwise separable blocks
    # (strides=2 causes spatial downsampling)
    # fmt: off
    ds_config = [
        # (pointwise_filters, strides)
        (64,   1),   # 32×32 → 32×32
        (128,  2),   # 32×32 → 16×16  ← downsample
        (128,  1),   # 16×16 → 16×16
        (256,  2),   # 16×16 →  8×8   ← downsample
        (256,  1),   #  8×8  →  8×8
        (512,  2),   #  8×8  →  4×4   ← downsample
        (512,  1),   #  4×4  →  4×4
        (512,  1),
        (512,  1),
        (512,  1),
        (512,  1),   # (5 blocks at 4×4×512 — feature refinement)
        (1024, 1),   #  4×4  →  4×4
        (1024, 1),   #  4×4  →  4×4
    ]
    # fmt: on

    for pw_filters, strides in ds_config:
        x = depthwise_separable_block(x, _apply_alpha(pw_filters, alpha), strides)

    # Classifier head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.40, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)

    model = tf.keras.Model(inputs, outputs, name=f"MobileNetV1_alpha{alpha}")
    return model


if __name__ == "__main__":
    model = build_mobilenet_v1()
    model.summary(line_length=80)
