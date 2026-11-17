# Neuron Vision — Neural Network from Scratch

An interpretable **64 → 32 → 10 multilayer perceptron (MLP)** for handwritten-digit recognition. Built from scratch with **NumPy** (no TensorFlow or PyTorch), with interactive hidden-neuron inspection in Google Colab.

## Results

| Metric | Value |
| --- | ---: |
| Held-out test accuracy | **98.15%** |
| Test samples | 270 |
| Train / validation samples | 1,257 / 270 |
| Training epochs | 150 |
| Dataset | scikit-learn Digits (8×8 images) |
| Network | 64 inputs → 32 tanh hidden → 10 softmax outputs |
| Loss / optimization | Cross-entropy / mini-batch SGD |

These results are from a reproducible notebook run (seed = 42). The evaluation set is held out from training.

## What's implemented

- Forward pass with **tanh** hidden units and numerically stable **softmax**
- Cross-entropy loss and vectorized **backpropagation**
- Mini-batch gradient descent
- Stratified train / validation / test splits
- Analytical gradients checked against finite differences
- Interactive neuron inspection: input pixels, selected neuron's learned weights, hidden activations and output class probabilities
- Learning-animation cell that records weight maps and activations across epochs

## Run locally

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python -m pytest -q tests
python -m src.train
```

Training writes artifacts into `outputs/`: a saved model, metrics, training curves and learned hidden-unit weight maps.

## Interactive Colab visualization

Open `neuron_vision_colab.ipynb` in Google Colab and **Runtime → Run all**. The notebook includes controls for choosing a handwritten digit and hidden neuron; its animation cell generates `outputs/neuron_learning.gif`.

The plot of a hidden neuron's 8×8 weight matrix is a **learned feature visualization**, *not* a generated digit image or a visualization of the neuron's internal physical structure.

## Repository layout

```text
src/
  __init__.py
  neural_network.py
  train.py
tests/
  test_neural_network.py
.github/workflows/tests.yml
README.md
requirements.txt
.gitignore
```

The Colab notebook and exported visualizations can be added alongside these source files.

## Limitations

This is a small 8×8 digit-classification educational model, **not a convolutional neural network**. The dataset is modest and test performance should not be generalized to real-world handwriting.

## Tech stack

Python · NumPy · scikit-learn · Matplotlib · pytest · ipywidgets · Google Colab
