import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { softmax, infer } from '../src/utils/inference.mjs';
import { grayscalePixels } from '../src/utils/preprocess.mjs';

test('stable softmax normalizes even large logits', () => {
  const p = softmax([1500, 1499, 1502]);
  assert.ok(p.every((n) => Number.isFinite(n) && n > 0 && n < 1));
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
});

test('8×8 grayscale values stay normalized', () => {
  const rgba = new Uint8ClampedArray(8 * 8 * 4);
  rgba[0] = rgba[1] = rgba[2] = 255;
  rgba[3] = 255;
  const pixels = grayscalePixels(rgba);
  assert.equal(pixels.length, 64);
  assert.equal(pixels[0], 1);
  assert.equal(pixels[1], 0);
});

test('exported model exactly matches Python predictions on held-out examples', () => {
  const modelFile = new URL('../public/model.json', import.meta.url);
  const examplesFile = new URL('../public/examples.json', import.meta.url);
  assert.ok(existsSync(modelFile), 'Run python -m src.export_web_model');
  assert.ok(existsSync(examplesFile), 'Run python -m src.export_web_model');
  const model = JSON.parse(readFileSync(modelFile, 'utf8'));
  const examples = JSON.parse(readFileSync(examplesFile, 'utf8')).examples;
  assert.equal(examples.length, 10);
  for (const sample of examples) {
    const actual = infer(sample.pixels, model);
    assert.equal(actual.prediction, sample.prediction);
    for (let i = 0; i < 10; i++) {
      assert.ok(
        Math.abs(actual.probabilities[i] - sample.probabilities[i]) < 1e-10,
        'Mismatch between NumPy and JavaScript inference'
      );
    }
  }
});
