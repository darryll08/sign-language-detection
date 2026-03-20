from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from src.config import (
    IMAGE_SIZE,
    NUM_CLASSES,
    LEARNING_RATE,
    MODEL_PATH,
)


def build_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    """
    Build model transfer learning dengan MobileNetV2.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


def compile_model(model):
    """
    Compile model.
    """
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_callbacks(model_path=MODEL_PATH):
    """
    Callback training.
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]
    return callbacks


def train_model(model, train_ds, val_ds, epochs):
    """
    Train model.
    """
    callbacks = get_callbacks()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    return history