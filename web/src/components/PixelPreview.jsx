export default function PixelPreview({ pixels }) {
  return (
    <div className="pixel-preview-wrap">
      <div className="pixel-grid" role="img" aria-label="The 8 by 8 grayscale pixels sent to the neural network">
        {Array.from({ length: 64 }, (_, i) => (
          <div
            key={i}
            className="pixel"
            style={{
              background: `rgb(${Math.round((pixels?.[i] ?? 0) * 255)} ${Math.round((pixels?.[i] ?? 0) * 255)} ${Math.round((pixels?.[i] ?? 0) * 255)})`
            }}
          />
        ))}
      </div>
      <div className="preview-meta">
        <strong>8 × 8</strong>
        <span>64 normalized pixels</span>
        <small>Actual model input</small>
      </div>
    </div>
  );
}
