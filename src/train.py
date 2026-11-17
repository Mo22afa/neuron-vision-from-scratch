import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from src.neural_network import NeuralNetwork


def load_data():
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target.astype(int)

    X_partial, X_test, y_partial, y_test = (
        train_test_split(
            X, y,
            test_size=0.15,
            stratify=y,
            random_state=42
        )
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_partial, y_partial,
            test_size=0.15 / 0.85,
            stratify=y_partial,
            random_state=42
        )
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    model = NeuralNetwork(seed=42)
    history = model.fit(X_train, y_train, X_val, y_val)
    test_loss, test_acc = model.evaluate(X_test, y_test)

    np.savez_compressed(
        out / "model.npz",
        W1=model.W1,
        b1=model.b1,
        W2=model.W2,
        b2=model.b2
    )

    metrics = {
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "test_loss": test_loss,
        "test_accuracy": test_acc
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("train", "val"):
        axes[0].plot(history[f"{name}_loss"], label=name)
        axes[1].plot(history[f"{name}_acc"], label=name)

    axes[0].set(title="Cross-entropy loss", xlabel="Epoch")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylim=(0, 1))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    scale = np.abs(model.W1).max()
    for k, ax in enumerate(axes.flat):
        ax.imshow(
            model.W1[:, k].reshape(8, 8),
            cmap="coolwarm",
            vmin=-scale,
            vmax=scale
        )
        ax.set_title(f"Neuron {k}", fontsize=8)
        ax.axis("off")

    fig.suptitle("Learned input weights of 32 hidden neurons")
    fig.tight_layout()
    fig.savefig(out / "neuron_weights.png", dpi=160)
    plt.close(fig)

    print(
        f"Train: {len(X_train)} | "
        f"Validation: {len(X_val)} | "
        f"Test: {len(X_test)}"
    )
    print(
        f"Test accuracy: {test_acc:.2%} | "
        f"Test loss: {test_loss:.4f}"
    )
    print("Results saved in outputs/")


if __name__ == "__main__":
    main()
