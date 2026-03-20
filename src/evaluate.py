from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.config import REPORTS_DIR, CLASS_NAMES


def plot_training_history(history, save_dir=REPORTS_DIR):
    """
    Plot accuracy dan loss dari history training.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    hist = history.history

    plt.figure(figsize=(10, 4))
    plt.plot(hist["accuracy"], label="train_accuracy")
    plt.plot(hist["val_accuracy"], label="val_accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "training_accuracy.png")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(hist["loss"], label="train_loss")
    plt.plot(hist["val_loss"], label="val_loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "training_loss.png")
    plt.show()


def get_true_and_pred_labels(model, dataset):
    """
    Ambil y_true dan y_pred dari dataset.
    """
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)

        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    return np.array(y_true), np.array(y_pred)


def show_classification_report(model, dataset, class_names=CLASS_NAMES):
    """
    Tampilkan classification report.
    """
    y_true, y_pred = get_true_and_pred_labels(model, dataset)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print(report)

    return y_true, y_pred


def plot_confusion_matrix(model, dataset, class_names=CLASS_NAMES, save_dir=REPORTS_DIR):
    """
    Plot confusion matrix.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = get_true_and_pred_labels(model, dataset)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(14, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png")
    plt.show()

    return cm