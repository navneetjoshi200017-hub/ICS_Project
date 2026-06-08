# Loads CIFAR-10 data
import tensorflow as tf
from config import BATCH_SIZE, USE_AUGMENTATION

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_cifar10():
    """
    Returns raw numpy arrays for train/test split.
    Labels are kept as integers (sparse) so SparseCategoricalCrossentropy
    can be used without one-hot encoding.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # y shape is (N, 1) — squeeze to (N,) for sparse loss
    y_train = y_train.squeeze()
    y_test  = y_test.squeeze()

    return (x_train, y_train), (x_test, y_test)


def _normalise(image, label):
    """Scale pixel values from [0, 255] to [0.0, 1.0]."""
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def _augment(image, label):
    """
    Stronger augmentation to make training harder and cap accuracy in 80-85%.
    More aggressive colour jitter + saturation jitter are the key additions.
    """
    image = tf.image.random_flip_left_right(image)
    # Pad by 4 pixels then random-crop back to 32×32
    image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    # Stronger colour jitter (was 0.1/0.9-1.1, now 0.2/0.7-1.3)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)  # new: hue/colour variance
    image = tf.image.random_hue(image, max_delta=0.05)               # new: slight hue shift
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def build_datasets(x_train, y_train, x_test, y_test):
    """
    Wraps numpy arrays into tf.data pipelines.

    Returns:
        train_ds  — shuffled, (optionally) augmented, batched
        val_ds    — last 10 % of training data, no augmentation
        test_ds   — test split, no augmentation
    """
    n_train = len(x_train)
    n_val   = int(n_train * 0.1)          # 10 % held-out validation

    # Split raw arrays
    x_val, y_val     = x_train[-n_val:],  y_train[-n_val:]
    x_tr,  y_tr      = x_train[:-n_val],  y_train[:-n_val]

    # Training pipeline
    train_ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
    train_ds = train_ds.shuffle(buffer_size=10_000, seed=42)
    train_ds = train_ds.map(_normalise,  num_parallel_calls=tf.data.AUTOTUNE)
    if USE_AUGMENTATION:
        train_ds = train_ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Validation pipeline (no augmentation)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.map(_normalise, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Test pipeline (no augmentation)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(_normalise, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds
