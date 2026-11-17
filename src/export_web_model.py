"""Export the exact NumPy model and held-out examples to the browser demo.

Run after: python -m src.train
"""
import json
from pathlib import Path

import numpy as np

from src.neural_network import NeuralNetwork
from src.train import load_data


def main():
    out = Path("web/public")
    out.mkdir(parents=True, exist_ok=True)
    source = Path("outputs/model.npz")
    if not source.exists():
        raise FileNotFoundError("Run python -m src.train first")

    model = NeuralNetwork()
    with np.load(source) as data:
        for key, shape in {
            "W1": (64, 32), "b1": (1, 32),
            "W2": (32, 10), "b2": (1, 10)
        }.items():
            arr = np.asarray(data[key], dtype=np.float64)
            if arr.shape != shape or not np.isfinite(arr).all():
                raise ValueError(f"Invalid {key} weights")
            setattr(model, key, arr)

    payload = {
        "architecture": [64, 32, 10],
        "dataset": "sklearn-digits-8x8",
        "normalization": "pixel_values_divided_by_16",
        "W1": model.W1.tolist(),
        "b1": model.b1.reshape(-1).tolist(),
        "W2": model.W2.tolist(),
        "b2": model.b2.reshape(-1).tolist()
    }
    (out / "model.json").write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )

    _, _, X_test, _, _, y_test = load_data()
    items = []
    for digit in range(10):
        index = int(np.flatnonzero(y_test == digit)[0])
        pixel_values = X_test[index]
        probabilities = model.forward(pixel_values[None, :])[1][0]
        items.append({
            "label": digit,
            "pixels": pixel_values.tolist(),
            "prediction": int(probabilities.argmax()),
            "probabilities": probabilities.tolist()
        })

    (out / "examples.json").write_text(
        json.dumps({"examples": items}, separators=(",", ":")),
        encoding="utf-8"
    )

    print(
        "Exported trained model (64→32→10) and "
        f"{len(items)} held-out examples to {out}"
    )


if __name__ == "__main__":
    main()
