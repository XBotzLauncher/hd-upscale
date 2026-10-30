import { InferenceSession, Tensor } from 'onnxruntime-node';
import sharp from 'sharp';
import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

/** Real-ESRGAN compact 4x, CPU via ONNX Runtime. */

const SCALE = 4;
const FORMATS = new Set(['jpeg', 'png', 'webp']);
export const DEFAULT_MAX_OUTPUT_PIXELS = 64_000_000;
const MODEL_PATH = fileURLToPath(new URL('../models/realesrgan-x4v3.onnx', import.meta.url));

function validateOptions(opts) {
  if (typeof opts !== 'object' || opts === null || Array.isArray(opts)) {
    throw new TypeError('opts must be an object');
  }
  const {
    tile = 512,
    overlap = 32,
    height,
    format = 'jpeg',
    quality = 95,
    maxOutputPixels = DEFAULT_MAX_OUTPUT_PIXELS,
  } = opts;

  if (!Number.isInteger(tile) || tile < 1) throw new RangeError('tile must be a positive integer');
  if (!Number.isInteger(overlap) || overlap < 0 || overlap >= tile) {
    throw new RangeError('overlap must be an integer >= 0 and < tile');
  }
  if (height !== undefined && (!Number.isInteger(height) || height < 1)) {
    throw new RangeError('height must be a positive integer');
  }
  if (!FORMATS.has(format)) throw new RangeError(`format must be one of: ${[...FORMATS].join(', ')}`);
  if (!Number.isInteger(quality) || quality < 1 || quality > 100) {
    throw new RangeError('quality must be an integer between 1 and 100');
  }
  if (maxOutputPixels !== Infinity && (!Number.isInteger(maxOutputPixels) || maxOutputPixels < 1)) {
    throw new RangeError('maxOutputPixels must be a positive integer or Infinity');
  }

  return { tile, overlap, height, format, quality, maxOutputPixels };
}

function checkPixels(w, h, max) {
  const out = w * SCALE * h * SCALE;
  if (out > max) {
    throw new RangeError(
      `output would be ${w * SCALE}x${h * SCALE} (${(out / 1e6).toFixed(1)} MP), above maxOutputPixels=${max}`,
    );
  }
}

let sessionPromise;
function getSession() {
  sessionPromise ??= InferenceSession.create(MODEL_PATH, { logSeverityLevel: 3 });
  return sessionPromise;
}

async function infer(session, rgb, w, h) {
  const inPlane = w * h;
  const input = new Float32Array(inPlane * 3);
  for (let p = 0; p < inPlane; p++) {
    for (let c = 0; c < 3; c++) input[c * inPlane + p] = rgb[p * 3 + c] / 255;
  }

  const { output } = await session.run({ input: new Tensor('float32', input, [1, 3, h, w]) });
  const data = output.data;
  const plane = w * SCALE * h * SCALE;
  const out = Buffer.allocUnsafe(plane * 3);
  for (let p = 0; p < plane; p++) {
    for (let c = 0; c < 3; c++) {
      const v = Math.round(data[c * plane + p] * 255);
      out[p * 3 + c] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  return out;
}

function crop(src, srcW, x, y, w, h) {
  const out = Buffer.allocUnsafe(w * h * 3);
  for (let row = 0; row < h; row++) {
    const start = ((y + row) * srcW + x) * 3;
    src.copy(out, row * w * 3, start, start + w * 3);
  }
  return out;
}

function blendPixel(res, si, dst, di, a) {
  const b = 1 - a;
  dst[di] = Math.round(res[si] * a + dst[di] * b);
  dst[di + 1] = Math.round(res[si + 1] * a + dst[di + 1] * b);
  dst[di + 2] = Math.round(res[si + 2] * a + dst[di + 2] * b);
}

/**
 * Paste a tile, cross-fading over the previous content on its left/top overlap.
 * fx/fy are feather widths in output pixels; 0 means no neighbor on that side.
 */
function writeTile(res, resW, dst, dstW, dx, dy, w, h, fx, fy) {
  const left = fx && fx < w ? fx : 0;
  for (let row = 0; row < h; row++) {
    const wy = fy && row < fy ? row / fy : 1;
    const si = row * resW * 3;
    const di = ((dy + row) * dstW + dx) * 3;
    if (wy >= 1) {
      for (let col = 0; col < left; col++) blendPixel(res, si + col * 3, dst, di + col * 3, col / left);
      if (left < w) res.copy(dst, di + left * 3, si + left * 3, si + w * 3);
    } else {
      for (let col = 0; col < w; col++) {
        blendPixel(res, si + col * 3, dst, di + col * 3, wy * (col < left ? col / left : 1));
      }
    }
  }
}

/**
 * Upscale an image 4x with Real-ESRGAN.
 *
 * @param {Buffer|string} input  image buffer or file path
 * @param {object} [opts]
 * @param {number} [opts.tile=512]    inference tile size in source pixels
 * @param {number} [opts.overlap=32]  tile overlap; hides seams, costs time
 * @param {number} [opts.height]      downscale result to this height (lanczos)
 * @param {'jpeg'|'png'|'webp'} [opts.format='jpeg']
 * @param {number} [opts.quality=95]
 * @param {number} [opts.maxOutputPixels=64000000]  cap on 4x output pixels (w*h*16); Infinity disables
 * @returns {Promise<Buffer>}
 */
export async function upscale(input, opts = {}) {
  const { tile, overlap, height, format, quality, maxOutputPixels } = validateOptions(opts);

  const meta = await sharp(input, { failOn: 'none' }).rotate().metadata();
  if (!meta.width || !meta.height) throw new Error('unsupported image');
  const swapped = meta.orientation >= 5 && meta.orientation <= 8;
  checkPixels(swapped ? meta.height : meta.width, swapped ? meta.width : meta.height, maxOutputPixels);

  const { data: rgb, info } = await sharp(input, { failOn: 'none' })
    .rotate()
    .removeAlpha()
    .toColourspace('srgb')
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { width: w, height: h } = info;
  checkPixels(w, h, maxOutputPixels);
  const ow = w * SCALE;
  const oh = h * SCALE;
  const out = Buffer.allocUnsafe(ow * oh * 3);
  const session = await getSession();

  for (let y = 0; y < h; y += tile) {
    for (let x = 0; x < w; x += tile) {
      const x0 = Math.max(0, x - overlap);
      const y0 = Math.max(0, y - overlap);
      const tw = Math.min(w, x + tile + overlap) - x0;
      const th = Math.min(h, y + tile + overlap) - y0;
      const patch = crop(rgb, w, x0, y0, tw, th);
      const res = await infer(session, patch, tw, th);
      const fx = (x - x0) * SCALE;
      const fy = (y - y0) * SCALE;
      writeTile(res, tw * SCALE, out, ow, x0 * SCALE, y0 * SCALE, tw * SCALE, th * SCALE, fx, fy);
    }
  }

  let img = sharp(out, { raw: { width: ow, height: oh, channels: 3 } });

  if (meta.hasAlpha) {
    const alpha = await sharp(input, { failOn: 'none' })
      .rotate()
      .toColourspace('srgb')
      .ensureAlpha()
      .extractChannel(3)
      .resize(ow, oh, { kernel: 'lanczos3' })
      .raw()
      .toBuffer();
    img = img.joinChannel(alpha, { raw: { width: ow, height: oh, channels: 1 } });
  }

  if (height && height < oh) {
    img = img.resize({ height, kernel: 'lanczos3' });
  }

  switch (format) {
    case 'jpeg':
      return img.jpeg({ quality, chromaSubsampling: '4:4:4' }).toBuffer();
    case 'png':
      return img.png({ compressionLevel: 9 }).toBuffer();
    case 'webp':
      return img.webp({ quality, effort: 6 }).toBuffer();
    default:
      throw new Error(`unsupported format: ${format}`);
  }
}

/** Upscale a file to disk. Returns output metadata. */
export async function upscaleFile(src, dst, opts = {}) {
  const buffer = await upscale(src, opts);
  await writeFile(dst, buffer);
  return sharp(buffer).metadata();
}
