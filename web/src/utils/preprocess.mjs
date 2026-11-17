/** Extract normalized luminance from the RGBA pixels of an 8×8 canvas. */
export function grayscalePixels(data) {
  if (data.length !== 8 * 8 * 4) {
    throw new Error('8×8 RGBA image required');
  }
  const pixels = [];
  for (let i = 0; i < data.length; i += 4) {
    const value = (data[i] * 0.2126 + data[i + 1] * 0.7152 +
      data[i + 2] * 0.0722) / 255;
    pixels.push(Math.max(0, Math.min(1, value)));
  }
  return pixels;
}

/**
 * Crop the ink bounding box, center in a square with margins, then downsample
 * with antialiasing to the model's native 8×8 input. Canvas uses white ink on
 * black, matching the polarity of sklearn's digits. Drawing accuracy is not
 * guaranteed to match the held-out Digits dataset.
 */
export function preprocessCanvas(canvas) {
  const context = canvas.getContext('2d', { willReadFrequently: true });
  const { width, height } = canvas;
  const image = context.getImageData(0, 0, width, height);
  let left = width, right = -1, top = height, bottom = -1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      if (Math.max(image.data[i], image.data[i + 1], image.data[i + 2]) > 22) {
        left = Math.min(left, x);
        right = Math.max(right, x);
        top = Math.min(top, y);
        bottom = Math.max(bottom, y);
      }
    }
  }
  if (right < left) return null;

  // An intermediate square prevents aliasing when shrinking to eight pixels.
  const workspace = document.createElement('canvas');
  workspace.width = workspace.height = 64;
  const work = workspace.getContext('2d');
  work.fillStyle = '#000';
  work.fillRect(0, 0, 64, 64);
  const w = right - left + 1;
  const h = bottom - top + 1;
  const targetSize = 48;
  const scale = targetSize / Math.max(w, h);
  work.imageSmoothingEnabled = true;
  work.imageSmoothingQuality = 'high';
  work.drawImage(
    canvas, left, top, w, h,
    (64 - w * scale) / 2, (64 - h * scale) / 2,
    w * scale, h * scale
  );

  const small = document.createElement('canvas');
  small.width = small.height = 8;
  const smallContext = small.getContext('2d', { willReadFrequently: true });
  smallContext.fillStyle = '#000';
  smallContext.fillRect(0, 0, 8, 8);
  smallContext.imageSmoothingEnabled = true;
  smallContext.imageSmoothingQuality = 'high';
  smallContext.drawImage(workspace, 0, 0, 8, 8);
  return grayscalePixels(smallContext.getImageData(0, 0, 8, 8).data);
}
