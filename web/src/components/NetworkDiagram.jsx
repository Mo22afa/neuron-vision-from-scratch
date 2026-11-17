import { useMemo, useState } from 'react';

const WIDTH = 1240;
const HEIGHT = 480;
const inputPoint = (i) => [130 + (i % 8) * 21, 132 + Math.floor(i / 8) * 35];
const hiddenPoint = (j) => [617, 77 + j * (333 / 31)];
const outputPoint = (k) => [1105, 90 + k * (320 / 9)];

function connection(a, b) {
  const mid = (a[0] + b[0]) / 2;
  return `M ${a[0]} ${a[1]} C ${mid} ${a[1]}, ${mid} ${b[1]}, ${b[0]} ${b[1]}`;
}

function strongestIndices(count, number, getValue) {
  return Array.from({ length: count }, (_, i) => i)
    .sort((a, b) => Math.abs(getValue(b)) - Math.abs(getValue(a)))
    .slice(0, number);
}

export default function NetworkDiagram({ pixels, result, model }) {
  const [focus, setFocus] = useState(15);
  const [animate, setAnimate] = useState(true);
  const hidden = result?.hidden ?? Array(32).fill(0);
  const probabilities = result?.probabilities ?? Array(10).fill(0);
  const live = Boolean(result && pixels && model);

  const edges = useMemo(() => {
    if (!model) return [];
    const selected = [];
    for (let neuron = 0; neuron < 32; neuron++) {
      const indices = strongestIndices(
        64, 3,
        (i) => live ? (pixels[i] * model.W1[i][neuron]) : model.W1[i][neuron]
      );
      for (const i of indices) {
        const weight = model.W1[i][neuron];
        const strength = live ? Math.abs(pixels[i] * weight) : 0;
        selected.push({
          id: `ih-${i}-${neuron}`,
          d: connection(inputPoint(i), hiddenPoint(neuron)),
          positive: weight > 0,
          strength: Math.min(1, strength / 1.3),
          active: live && strength > 0.025,
        });
      }
    }
    for (let digit = 0; digit < 10; digit++) {
      const indices = strongestIndices(
        32, 4,
        (j) => live ? hidden[j] * model.W2[j][digit] : model.W2[j][digit]
      );
      for (const j of indices) {
        const weight = hidden[j] * model.W2[j][digit];
        const strength = live ? Math.abs(weight) : 0;
        selected.push({
          id: `ho-${j}-${digit}`,
          d: connection(hiddenPoint(j), outputPoint(digit)),
          positive: weight > 0,
          strength: Math.min(1, strength / 2.4),
          active: live && strength > 0.04,
        });
      }
    }
    return selected;
  }, [pixels, hidden, model, live]);

  return (
    <section className="glass network-panel" id="network">
      <div className="panel-top network-panel-heading">
        <div>
          <p className="eyebrow">03 / INSIDE THE NETWORK</p>
          <h2>See the signal flow</h2>
          <p className="supporting-text">Real 64 → 32 → 10 neuron values and selected strong connections.</p>
        </div>
        <button
          className={animate ? 'soft-button enabled' : 'soft-button'}
          type="button"
          onClick={() => setAnimate(!animate)}
        >
          <span className="play-icon">{animate ? 'Ⅱ' : '▶'}</span>
          {animate ? 'Pause flow' : 'Play flow'}
        </button>
      </div>
      <div className="network-viewport">
        <svg className={animate ? 'network-svg running' : 'network-svg'} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Real NumPy neural network architecture diagram with 64 input neurons, 32 hidden neurons, and 10 output neurons.">
          <defs>
            <linearGradient id="network-bg" x1="0" x2="1" y1="0" y2="1">
              <stop offset="0%" stopColor="#091524" />
              <stop offset="100%" stopColor="#101029" />
            </linearGradient>
            <filter id="node-glow" x="-100%" y="-100%" width="300%" height="300%">
              <feGaussianBlur stdDeviation="5" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
          <rect width={WIDTH} height={HEIGHT} rx="17" fill="url(#network-bg)" />
          <g className="network-grid">
            {Array.from({ length: 25 }, (_, i) => (
              <path key={`v${i}`} d={`M ${i * 52} 0 V480`} />
            ))}
            {Array.from({ length: 10 }, (_, i) => (
              <path key={`h${i}`} d={`M 0 ${i * 52} H1240`} />
            ))}
          </g>
          <g className="layer-headings">
            <text x="204" y="45" textAnchor="middle">INPUT · 64</text>
            <text x="617" y="45" textAnchor="middle">HIDDEN · 32</text>
            <text x="1105" y="45" textAnchor="middle">OUTPUT · 10</text>
          </g>

          <g className="network-connections">
            {edges.map((edge, index) => (
              <path
                key={edge.id}
                d={edge.d}
                fill="none"
                className={edge.active ? 'signal-edge active-edge' : 'signal-edge'}
                stroke={edge.positive ? '#43e9e5' : '#a276ff'}
                strokeWidth={edge.active ? 0.9 + edge.strength * 2 : 0.6}
                opacity={edge.active ? 0.18 + edge.strength * 0.7 : 0.085}
                style={{ animationDelay: `${(index % 14) * -0.17}s` }}
              />
            ))}
          </g>

          <g className="input-layer">
            {Array.from({ length: 64 }, (_, i) => {
              const [x, y] = inputPoint(i);
              const value = pixels?.[i] ?? 0;
              return (
                <circle key={i} cx={x} cy={y} r={3 + value * 3.2}
                  fill={value > 0.1 ? '#70f5ed' : '#294865'}
                  stroke={value > 0.3 ? '#ccfffc' : '#37607e'}
                  strokeWidth="0.8" opacity={0.47 + value * 0.53}>
                  <title>{`Input pixel ${i}: ${value.toFixed(3)}`}</title>
                </circle>
              );
            })}
          </g>

          <g className="hidden-layer">
            {Array.from({ length: 32 }, (_, j) => {
              const [x, y] = hiddenPoint(j);
              const activation = hidden[j] ?? 0;
              const isFocused = focus === j;
              return (
                <g key={j} className="node-interactive" tabIndex={0}
                  role="button" aria-label={`Inspect hidden neuron ${j}`}
                  onMouseEnter={() => setFocus(j)}
                  onFocus={() => setFocus(j)}
                  onClick={() => setFocus(j)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      setFocus(j);
                    }
                  }}>
                  {isFocused && <circle cx={x} cy={y} r="14" fill="none" stroke="#49f5e7" strokeWidth="1.3" opacity=".95" />}
                  <circle cx={x} cy={y} r={isFocused ? 7.7 : 5.5 + Math.abs(activation) * 2.2}
                    fill={!live ? '#1e4358' : activation >= 0 ? '#36d9c7' : '#9171f6'}
                    stroke={isFocused ? '#b4fff8' : '#446987'} strokeWidth="1.3"
                    opacity={live ? 0.42 + Math.abs(activation) * .58 : .58}
                    filter={isFocused || Math.abs(activation) > .7 ? 'url(#node-glow)' : undefined}>
                    <title>{`Hidden neuron ${j}: ${activation.toFixed(3)}`}</title>
                  </circle>
                </g>
              );
            })}
          </g>

          <g className="output-layer">
            {Array.from({ length: 10 }, (_, k) => {
              const [x, y] = outputPoint(k);
              const p = probabilities[k] || 0;
              const winner = result && result.prediction === k;
              return (
                <g key={k}>
                  {winner && <circle cx={x} cy={y} r="19" fill="#ff7955" opacity=".16" filter="url(#node-glow)" />}
                  <circle cx={x} cy={y} r={winner ? 11 : 7}
                    fill={winner ? '#ff8d62' : '#232b48'}
                    stroke={winner ? '#ffd2ab' : '#637192'}
                    strokeWidth={winner ? 2.3 : 1.2} />
                  <text x={x + 25} y={y + 5} className={winner ? 'output-digit winner-text' : 'output-digit'}>
                    {k} <tspan className="prob-inline">{result ? `${Math.round(p * 100)}%` : ''}</tspan>
                  </text>
                </g>
              );
            })}
          </g>
          <text x="204" y="448" className="layer-subtitle" textAnchor="middle">8 × 8 grayscale</text>
          <text x="617" y="448" className="layer-subtitle" textAnchor="middle">tanh activation</text>
          <text x="1105" y="448" className="layer-subtitle" textAnchor="middle">softmax</text>
        </svg>
      </div>
      <div className="network-footer">
        <div className="legend">
          <span><i className="legend-dot teal" /> Positive connection</span>
          <span><i className="legend-dot violet" /> Negative connection</span>
          <span><i className="legend-dot orange" /> Predicted class</span>
        </div>
        <p>Inspect neuron <strong>#{focus}</strong> · activation <strong>{hidden[focus].toFixed(3)}</strong></p>
      </div>
      <p className="network-disclaimer">Only selected high-contribution connections are drawn for clarity. All 64 inputs, 32 hidden units, and 10 outputs are computed.</p>
    </section>
  );
}
