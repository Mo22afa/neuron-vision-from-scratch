import { useCallback, useEffect, useMemo, useState } from 'react';
import DrawingCanvas from './components/DrawingCanvas.jsx';
import PredictionPanel from './components/PredictionPanel.jsx';
import PixelPreview from './components/PixelPreview.jsx';
import NetworkDiagram from './components/NetworkDiagram.jsx';
import { infer } from './utils/inference.mjs';

const REPOSITORY = 'https://github.com/Mo22afa/neuron-vision-from-scratch';

export default function App() {
  const [model, setModel] = useState(null);
  const [examples, setExamples] = useState([]);
  const [error, setError] = useState('');
  const [pixels, setPixels] = useState(null);
  const [sample, setSample] = useState(null);
  const [clearVersion, setClearVersion] = useState(0);
  const [source, setSource] = useState('draw');

  useEffect(() => {
    let cancelled = false;
    const base = import.meta.env.BASE_URL;
    Promise.all([
      fetch(`${base}model.json`),
      fetch(`${base}examples.json`),
    ]).then(async ([weights, examplesResponse]) => {
      if (!weights.ok || !examplesResponse.ok) {
        throw new Error('Unable to load trained model files.');
      }
      return Promise.all([weights.json(), examplesResponse.json()]);
    }).then(([trainedModel, sampleData]) => {
      if (cancelled) return;
      if (trainedModel.architecture?.join(',') !== '64,32,10') {
        throw new Error('Wrong model architecture. Expected 64→32→10.');
      }
      setModel(trainedModel);
      setExamples(sampleData.examples ?? []);
    }).catch((failure) => {
      if (!cancelled) setError(failure.message);
    });
    return () => { cancelled = true; };
  }, []);

  const result = useMemo(() => {
    if (!model || !pixels) return null;
    return infer(pixels, model);
  }, [model, pixels]);

  const onDraw = useCallback((nextPixels) => {
    setPixels(nextPixels);
    setSource('draw');
  }, []);

  function clearDrawing() {
    setSample(null);
    setPixels(null);
    setSource('draw');
    setClearVersion((count) => count + 1);
  }

  function loadExample(digit) {
    const example = examples.find((item) => item.label === digit);
    if (!example) return;
    setPixels(example.pixels);
    setSample({ ...example });
    setSource(`example ${digit}`);
  }

  return (
    <div className="app-shell">
      <div className="ambient ambient-cyan" aria-hidden="true" />
      <div className="ambient ambient-violet" aria-hidden="true" />

      <header className="site-header">
        <a href="#top" className="brand" aria-label="Neuron Vision home">
          <span className="brand-symbol" aria-hidden="true"><span /><span /><span /><span /></span>
          <span>NEURON<span className="brand-accent">VISION</span></span>
        </a>
        <div className="header-actions">
          <span className="header-tag">BUILT FROM SCRATCH</span>
          <a className="header-github" href={REPOSITORY} target="_blank" rel="noreferrer">
            <svg aria-hidden="true" width="17" height="17" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 .5A11.5 11.5 0 0 0 8.36 23c.58.11.79-.25.79-.56v-2.13c-3.22.7-3.9-1.37-3.9-1.37-.53-1.34-1.3-1.7-1.3-1.7-1.05-.72.08-.71.08-.71 1.16.08 1.77 1.19 1.77 1.19 1.03 1.76 2.7 1.25 3.36.95.11-.75.4-1.25.73-1.54-2.57-.29-5.28-1.29-5.28-5.69 0-1.25.45-2.27 1.19-3.07-.12-.29-.52-1.46.11-3.04 0 0 .97-.31 3.16 1.17A10.9 10.9 0 0 1 12 6.41a10.9 10.9 0 0 1 2.88.39c2.19-1.48 3.15-1.17 3.15-1.17.63 1.58.23 2.75.12 3.04.74.8 1.18 1.82 1.18 3.07 0 4.41-2.71 5.4-5.29 5.69.41.35.78 1.04.78 2.1v2.91c0 .31.21.67.79.56A11.5 11.5 0 0 0 12 .5Z" />
            </svg>
            VIEW ON GITHUB <span aria-hidden="true">↗</span>
          </a>
        </div>
      </header>

      <main id="top" className="main-content">
        <section className="hero-section">
          <div className="hero-eyebrow"><span className="spark" /> INTERACTIVE MACHINE LEARNING LAB <span className="hero-line" /></div>
          <h1>Draw a digit. <span>Watch the<br className="desktop-break" /> network think.</span></h1>
          <p>Sketch a number from 0–9 and see what happens inside a real neural network — every activation, every probability, all calculated live.</p>
          <div className="hero-stats">
            <div><strong>98.15<span>%</span></strong><span>TEST ACCURACY</span></div>
            <span className="stat-separator" />
            <div><strong>64<span>→</span>32<span>→</span>10</strong><span>NETWORK ARCHITECTURE</span></div>
            <span className="stat-separator" />
            <div><strong>0</strong><span>DEEP LEARNING FRAMEWORKS</span></div>
          </div>
        </section>

        {error && (
          <div role="alert" className="model-error">
            Could not load the trained model: {error}. Check the model export and reload.
          </div>
        )}

        <div className="dashboard-layout">
          <PredictionPanel result={result} ready={Boolean(model)} />

          <section className="glass drawing-panel">
            <div className="panel-top">
              <div>
                <p className="eyebrow">02 / SKETCHPAD</p>
                <h2>Your turn to draw</h2>
              </div>
              <span className={model ? 'live-label' : 'live-label offline'}>
                <span className="status-light" /> {model ? 'LIVE INFERENCE' : 'INITIALIZING'}
              </span>
            </div>
            <p className="supporting-text">White ink on black. Draw with your mouse or finger.</p>
            <DrawingCanvas onDraw={onDraw} sample={sample} clearVersion={clearVersion} disabled={!model} />
            <div className="drawing-actions">
              <button className="primary-button" onClick={clearDrawing} type="button">
                <span aria-hidden="true">↺</span> Clear canvas
              </button>
              <span className="draw-hint">TRY ANY DIGIT 0–9</span>
            </div>
            <div className="sample-picker">
              <div className="sample-heading"><strong>Or try a real test sample</strong><span>Held-out dataset</span></div>
              <div className="sample-options">
                {Array.from({ length: 10 }, (_, digit) => (
                  <button
                    key={digit}
                    type="button"
                    title={`Load held-out handwritten digit ${digit}`}
                    disabled={!model || !examples.length}
                    className={source === `example ${digit}` ? 'sample-button chosen' : 'sample-button'}
                    onClick={() => loadExample(digit)}>
                    {digit}
                  </button>
                ))}
              </div>
            </div>
          </section>

          <aside className="glass input-panel">
            <div>
              <p className="eyebrow">INPUT MONITOR</p>
              <h2>Through the lens</h2>
              <p className="supporting-text">What the network actually receives.</p>
            </div>
            <PixelPreview pixels={pixels} />
            <div className="input-insight">
              <span className="mini-heading">DATA FLOW</span>
              <div className="flow-item"><span>01</span><div><strong>Raw sketch</strong><p>320 × 320 canvas</p></div></div>
              <div className="flow-line" />
              <div className="flow-item"><span>02</span><div><strong>Preprocess</strong><p>Crop · center · downsample</p></div></div>
              <div className="flow-line" />
              <div className="flow-item"><span>03</span><div><strong>Forward pass</strong><p>NumPy weights in JavaScript</p></div></div>
            </div>
            <div className="model-note"><span className="tiny-dot" /> Actual 8×8 scikit-learn Digits model, not a CNN.</div>
          </aside>
        </div>

        <NetworkDiagram pixels={pixels} result={result} model={model} />

        <section className="explain-section" id="about">
          <div>
            <p className="eyebrow">WHAT'S UNDER THE HOOD</p>
            <h2>Not a simulation.<br /><span>The real model.</span></h2>
            <p>A from-scratch multilayer perceptron trained with NumPy on the scikit-learn Digits dataset. The browser runs exactly the same weights, tanh activations and softmax inference as the Python version.</p>
            <a href={`${REPOSITORY}/blob/main/src/neural_network.py`} target="_blank" rel="noreferrer" className="code-link">EXPLORE THE SOURCE <span aria-hidden="true">↗</span></a>
          </div>
          <div className="feature-list">
            <div><span className="feature-icon">01</span><div><strong>Built with mathematics</strong><p>Forward pass, cross-entropy, backpropagation and gradient descent — implemented without TensorFlow or PyTorch.</p></div></div>
            <div><span className="feature-icon">02</span><div><strong>Inspectable hidden neurons</strong><p>Select any of the 32 hidden neurons and watch its actual tanh activation change as the input changes.</p></div></div>
            <div><span className="feature-icon">03</span><div><strong>Verified predictions</strong><p>JavaScript probabilities are checked against the trained Python model on held-out examples.</p></div></div>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <a className="brand footer-brand" href="#top">NEURON<span className="brand-accent">VISION</span></a>
        <span>Made to make neural networks understandable.</span>
        <a href={REPOSITORY} target="_blank" rel="noreferrer">SOURCE CODE ↗</a>
      </footer>
    </div>
  );
}
