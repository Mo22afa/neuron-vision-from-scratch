import { useEffect, useRef } from 'react';
import { preprocessCanvas } from '../utils/preprocess.mjs';

const CANVAS_SIZE = 320;

function fillBlack(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 23;
}

export default function DrawingCanvas({ onDraw, sample, clearVersion, disabled }) {
  const canvasRef = useRef(null);
  const drawingRef = useRef(false);
  const sampleModeRef = useRef(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    fillBlack(canvas);
  }, [clearVersion]);

  useEffect(() => {
    if (!sample) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    fillBlack(canvas);
    const small = document.createElement('canvas');
    small.width = small.height = 8;
    const sctx = small.getContext('2d');
    const image = sctx.createImageData(8, 8);
    sample.pixels.forEach((value, i) => {
      const intensity = Math.round(Math.max(0, Math.min(1, value)) * 255);
      image.data[4 * i] = intensity;
      image.data[4 * i + 1] = intensity;
      image.data[4 * i + 2] = intensity;
      image.data[4 * i + 3] = 255;
    });
    sctx.putImageData(image, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
    sampleModeRef.current = true;
  }, [sample]);

  function position(event) {
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * CANVAS_SIZE / rect.width,
      y: (event.clientY - rect.top) * CANVAS_SIZE / rect.height
    };
  }

  function emit() {
    onDraw(preprocessCanvas(canvasRef.current));
  }

  function start(event) {
    if (disabled) return;
    event.preventDefault();
    const canvas = canvasRef.current;
    if (sampleModeRef.current) {
      fillBlack(canvas);
      sampleModeRef.current = false;
    }
    canvas.setPointerCapture(event.pointerId);
    drawingRef.current = true;
    const ctx = canvas.getContext('2d');
    const { x, y } = position(event);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 0.01, y + 0.01);
    ctx.stroke();
    emit();
  }

  function move(event) {
    if (!drawingRef.current || disabled) return;
    event.preventDefault();
    const ctx = canvasRef.current.getContext('2d');
    const { x, y } = position(event);
    ctx.lineTo(x, y);
    ctx.stroke();
    emit();
  }

  function end(event) {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    if (canvasRef.current.hasPointerCapture(event.pointerId)) {
      canvasRef.current.releasePointerCapture(event.pointerId);
    }
  }

  return (
    <div className="canvas-shell">
      <canvas
        ref={canvasRef}
        className="draw-canvas"
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        aria-label="Drawing area. Use your finger or mouse to draw a digit from zero to nine."
        onPointerDown={start}
        onPointerMove={move}
        onPointerUp={end}
        onPointerCancel={end}
        onLostPointerCapture={() => { drawingRef.current = false; }}
      />
      <div className="canvas-corner corner-tl" />
      <div className="canvas-corner corner-tr" />
      <div className="canvas-corner corner-bl" />
      <div className="canvas-corner corner-br" />
    </div>
  );
}
