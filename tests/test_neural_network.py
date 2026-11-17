import numpy as np

from src.neural_network import NeuralNetwork


def test_probabilities():
    model = NeuralNetwork(input_size=4, hidden_size=5, output_size=3)
    X = np.ones((6, 4))
    p = model.predict_proba(X)
    assert p.shape == (6, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)


def test_backpropagation_against_finite_differences():
    model = NeuralNetwork(input_size=4, hidden_size=3, output_size=2)
    X = np.array([[0.2, 0.8, 0.1, 0.3], [0.5, 0.0, 0.9, 0.1]])
    y = np.array([0, 1])
    _, grads = model.loss_and_grads(X, y)
    epsilon = 1e-5

    for name in ("W1", "b1", "W2", "b2"):
        param = getattr(model, name)
        for idx in np.ndindex(param.shape):
            original = param[idx]
            param[idx] = original + epsilon
            plus = model.loss_and_grads(X, y)[0]
            param[idx] = original - epsilon
            minus = model.loss_and_grads(X, y)[0]
            param[idx] = original
            numerical = (plus - minus) / (2 * epsilon)
            np.testing.assert_allclose(
                numerical, grads[name][idx],
                rtol=1e-4, atol=1e-7
            )


def test_training_decreases_loss():
    model = NeuralNetwork(input_size=4, hidden_size=6, output_size=2)
    rng = np.random.default_rng(3)
    X = rng.uniform(size=(100, 4))
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    before = model.evaluate(X, y)[0]
    model.fit(X, y, X, y, epochs=80, lr=0.4)
    after = model.evaluate(X, y)[0]
    assert after < before
