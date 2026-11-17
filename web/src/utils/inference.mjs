/** Equivalent to the NumPy inference used by src/neural_network.py. */
export function softmax(values) {
  if (!Array.isArray(values) || values.length === 0) {
    throw new Error('Softmax requires a non-empty array');
  }
  const max = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - max));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

export function infer(pixels, model) {
  if (!model || pixels.length !== 64 ||
      model.W1?.length !== 64 || model.W2?.length !== 32 ||
      model.b1?.length !== 32 || model.b2?.length !== 10) {
    throw new Error('Expected a 64→32→10 trained model and 64 input pixels');
  }
  const hidden = new Array(32);
  for (let j = 0; j < 32; j++) {
    let sum = model.b1[j];
    for (let i = 0; i < 64; i++) sum += pixels[i] * model.W1[i][j];
    hidden[j] = Math.tanh(sum);
  }
  const logits = new Array(10);
  for (let k = 0; k < 10; k++) {
    let sum = model.b2[k];
    for (let j = 0; j < 32; j++) sum += hidden[j] * model.W2[j][k];
    logits[k] = sum;
  }
  const probabilities = softmax(logits);
  const prediction = probabilities.indexOf(Math.max(...probabilities));
  return {
    hidden,
    logits,
    probabilities,
    prediction,
    confidence: probabilities[prediction]
  };
}
