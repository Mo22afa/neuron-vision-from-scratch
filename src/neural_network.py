import numpy as np


class NeuralNetwork:
    """64 -> 32 -> 10 MLP implemented with NumPy only."""

    def __init__(self, input_size=64, hidden_size=32,
                 output_size=10, seed=42):

        rng = np.random.default_rng(seed)

        self.W1 = rng.normal(
            0, 1 / np.sqrt(input_size),
            (input_size, hidden_size)
        )
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = rng.normal(
            0, 1 / np.sqrt(hidden_size),
            (hidden_size, output_size)
        )
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        hidden = np.tanh(X @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        exp_logits = np.exp(
            logits - logits.max(axis=1, keepdims=True)
        )
        probs = exp_logits / exp_logits.sum(
            axis=1, keepdims=True
        )
        return hidden, probs

    def loss_and_grads(self, X, y):
        hidden, probs = self.forward(X)
        n = len(y)

        loss = -np.log(
            np.clip(
                probs[np.arange(n), y], 1e-12, 1
            )
        ).mean()

        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1
        dlogits /= n

        dW2 = hidden.T @ dlogits
        db2 = dlogits.sum(axis=0, keepdims=True)

        dhidden = (
            dlogits @ self.W2.T
        ) * (1 - hidden ** 2)

        dW1 = X.T @ dhidden
        db1 = dhidden.sum(axis=0, keepdims=True)

        return loss, {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2
        }

    def predict_proba(self, X):
        return self.forward(X)[1]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def evaluate(self, X, y):
        probs = self.predict_proba(X)
        loss = -np.log(
            np.clip(
                probs[np.arange(len(y)), y],
                1e-12, 1
            )
        ).mean()

        accuracy = np.mean(
            probs.argmax(axis=1) == y
        )
        return float(loss), float(accuracy)

    def fit(self, X, y, X_val, y_val,
            epochs=150, batch_size=64,
            lr=0.35, seed=42):

        rng = np.random.default_rng(seed)

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        for _ in range(epochs):
            indices = rng.permutation(len(X))

            for start in range(0, len(X), batch_size):
                batch = indices[start:start + batch_size]
                _, grads = self.loss_and_grads(
                    X[batch], y[batch]
                )

                for name, grad in grads.items():
                    setattr(
                        self,
                        name,
                        getattr(self, name) - lr * grad
                    )

            for label, a, b in (
                ("train", X, y),
                ("val", X_val, y_val)
            ):
                loss, acc = self.evaluate(a, b)
                history[f"{label}_loss"].append(loss)
                history[f"{label}_acc"].append(acc)

        return history
