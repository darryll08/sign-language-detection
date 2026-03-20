from typing import Tuple

import tensorflow as tf

from src.config import IMAGE_SIZE, BATCH_SIZE, CLASS_NAMES, NUM_CLASSES

AUTOTUNE = tf.data.AUTOTUNE


label_to_index = {label: idx for idx, label in enumerate(CLASS_NAMES)}
index_to_label = {idx: label for idx, label in enumerate(CLASS_NAMES)}


def encode_label(label: tf.Tensor) -> tf.Tensor:
    """
    Ubah label string menjadi index integer.
    """
    keys = tf.constant(list(label_to_index.keys()))
    vals = tf.constant(list(label_to_index.values()), dtype=tf.int32)

    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(keys, vals),
        default_value=-1
    )
    return table.lookup(label)


def load_and_preprocess_image(filepath: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load image dari path, decode JPG, resize, normalize, encode label.
    """
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    label_index = encode_label(label)
    label_onehot = tf.one_hot(label_index, depth=NUM_CLASSES)

    return image, label_onehot


def augment_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Augmentasi ringan untuk train set.
    """
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


def make_dataset(
    df,
    batch_size: int = BATCH_SIZE,
    training: bool = False
) -> tf.data.Dataset:
    """
    Ubah dataframe menjadi tf.data.Dataset.
    """
    filepaths = df["filepath"].values
    labels = df["label"].values

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    ds = ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)

    return ds