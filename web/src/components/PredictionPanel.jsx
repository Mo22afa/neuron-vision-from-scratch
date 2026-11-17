export default function PredictionPanel({ result, ready }) {
  const probabilities = result?.probabilities ?? Array(10).fill(0);
  return (
    <section className="glass probabilities-panel" aria-label="Digit probability predictions">
      <div className="panel-top">
        <div>
          <p className="eyebrow">01 / LIVE PREDICTION</p>
          <h2>Model confidence</h2>
        </div>
        <span className={ready ? 'status-pill' : 'status-pill muted'}>
          <span className="status-light" />
          {ready ? 'MODEL ONLINE' : 'LOADING'}
        </span>
      </div>
      <p className="supporting-text">How confident is the network about each digit?</p>
      <div className="prob-grid">
        {Array.from({ length: 10 }, (_, digit) => {
          const probability = probabilities[digit];
          const selected = result && result.prediction === digit;
          return (
            <div className={selected ? 'prob-tile winning' : 'prob-tile'} key={digit}>
              <div className="prob-topline">
                <strong>{digit}</strong>
                <span>{result ? (probability * 100).toFixed(probability < 0.01 ? 1 : 0) : '—'}{result ? '%' : ''}</span>
              </div>
              <div className="prob-track">
                <div className="prob-fill" style={{ width: `${probability * 100}%` }} />
              </div>
            </div>
          );
        })}
      </div>
      <div className="prediction-summary">
        <div className="prediction-label">
          <span>TOP PREDICTION</span>
          <strong>{result ? `${Math.round(result.confidence * 100)}% confidence` : 'Draw to begin'}</strong>
        </div>
        <div className={result ? 'prediction-number active' : 'prediction-number'}>
          {result ? result.prediction : '—'}
        </div>
      </div>
      <p className="caution-text">98.15% is held-out test accuracy on 8×8 digits. Freehand drawings can be less accurate.</p>
    </section>
  );
}
