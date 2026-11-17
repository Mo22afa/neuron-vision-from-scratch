"""Render a reproducible GIF of learned weights and hidden-neuron activations.

Usage: python -m src.animate
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from src.neural_network import NeuralNetwork
from src.train import load_data


def collect_snapshots(epochs=150, every=5, seed=42):
    X_train, _, X_test, y_train, _, y_test = load_data()
    model = NeuralNetwork(seed=seed)
    rng = np.random.default_rng(seed)
    sample = X_test[0:1]
    truth = int(y_test[0])
    records = []

    def capture(epoch):
        hidden, probs = model.forward(sample)
        _, accuracy = model.evaluate(X_train, y_train)
        records.append((
            epoch, model.W1.copy(), hidden[0].copy(),
            probs[0].copy(), accuracy
        ))

    capture(0)
    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(len(X_train))
        for start in range(0, len(X_train), 64):
            batch = permutation[start:start + 64]
            _, grads = model.loss_and_grads(X_train[batch], y_train[batch])
            for name, grad in grads.items():
                setattr(model, name, getattr(model, name) - 0.35 * grad)
        if epoch % every == 0:
            capture(epoch)
    return records, sample, truth


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    snapshots, sample, true_digit = collect_snapshots()
    maximum = max(np.abs(item[1]).max() for item in snapshots)

    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(13, 6), facecolor="#0b1220")
        grid = fig.add_gridspec(
            4, 12, left=0.045, right=0.975,
            top=0.84, bottom=0.12, wspace=0.25, hspace=0.4
        )
        images = []
        for neuron in range(32):
            ax = fig.add_subplot(grid[neuron // 8, neuron % 8])
            img = ax.imshow(
                snapshots[0][1][:, neuron].reshape(8, 8),
                cmap="coolwarm", vmin=-maximum, vmax=maximum
            )
            ax.set_title(f"N{neuron:02d}", fontsize=8)
            ax.axis("off")
            images.append(img)

        input_ax = fig.add_subplot(grid[0:2, 8:10])
        input_ax.imshow(sample.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        input_ax.set_title(f"Input: digit {true_digit}", fontsize=11)
        input_ax.axis("off")

        hidden_ax = fig.add_subplot(grid[0:2, 10:12])
        activation_img = hidden_ax.imshow(
            snapshots[0][2].reshape(4, 8),
            cmap="coolwarm", vmin=-1, vmax=1
        )
        hidden_ax.set_title("Activations", fontsize=11)
        hidden_ax.axis("off")

        pred_ax = fig.add_subplot(grid[2:4, 8:12])
        bars = pred_ax.bar(range(10), snapshots[0][3], color="#64748b")
        pred_ax.set_xticks(range(10))
        pred_ax.set_ylim(0, 1)
        pred_ax.set_title("Output Probabilities")
        pred_ax.set_xlabel("Predicted Digit")
        pred_ax.set_ylabel("Probability")
        pred_ax.spines["top"].set_visible(False)
        pred_ax.spines["right"].set_visible(False)
        heading = fig.suptitle(
            "Neuron Learning | Epoch 0", fontsize=17, weight="bold", y=0.97
        )
        subtitle = fig.text(
            0.5, 0.895, "", ha="center", fontsize=11, color="#cbd5e1"
        )

        def update(index):
            epoch, weights, hidden, probs, accuracy = snapshots[index]
            for neuron, img in enumerate(images):
                img.set_data(weights[:, neuron].reshape(8, 8))
            activation_img.set_data(hidden.reshape(4, 8))
            predicted_digit = int(np.argmax(probs))
            for digit, (bar, probability) in enumerate(zip(bars, probs)):
                bar.set_height(float(probability))
                bar.set_color(
                    "#2dd4bf" if digit == predicted_digit else "#64748b"
                )
            heading.set_text(
                f"How 32 Hidden Neurons Learn | Epoch {epoch:03d}"
            )
            subtitle.set_text(
                f"Training Accuracy: {accuracy:.1%}   |   "
                f"True Digit: {true_digit}   |   Prediction: {predicted_digit}"
            )

        animation = FuncAnimation(
            fig, update, frames=len(snapshots),
            interval=160, blit=False, repeat=True
        )
        target = output_dir / "neuron_learning.gif"
        animation.save(str(target), writer=PillowWriter(fps=6), dpi=85)
        plt.close(fig)
        print(f"Saved {target} ({len(snapshots)} frames)")


if __name__ == "__main__":
    main()
