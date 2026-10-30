import { before, test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import { upscale } from '../src/index.js';

const FIXTURE = fileURLToPath(new URL('./fixtures/sample.png', import.meta.url));
const meta = (buf) => sharp(buf).metadata();

let full;
before(async () => {
  full = await upscale(FIXTURE, { format: 'png' });
});

test('upscales a real photo 4x', async () => {
  const m = await meta(full);
  assert.deepEqual([m.width, m.height, m.format], [480, 640, 'png']);
});

test('downscales the 4x result to a requested height', async () => {
  const m = await meta(await upscale(FIXTURE, { height: 320, format: 'jpeg' }));
  assert.deepEqual([m.width, m.height, m.format], [240, 320, 'jpeg']);
});

test('preserves alpha', async () => {
  const rgba = await sharp(FIXTURE).resize(48, 64).ensureAlpha(0.4).png().toBuffer();
  const m = await meta(await upscale(rgba, { format: 'png' }));
  assert.deepEqual([m.width, m.height, m.hasAlpha], [192, 256, true]);
});

test('encodes webp', async () => {
  const m = await meta(await upscale(FIXTURE, { height: 80, format: 'webp' }));
  assert.equal(m.format, 'webp');
  assert.equal(m.height, 80);
});

test('tiling with feathering matches a single-tile pass', async () => {
  const tiled = await upscale(FIXTURE, { tile: 48, overlap: 8, format: 'png' });
  const m = await meta(tiled);
  assert.deepEqual([m.width, m.height], [480, 640]);

  const a = await sharp(tiled).raw().toBuffer();
  const b = await sharp(full).raw().toBuffer();
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  const mean = sum / a.length;
  assert.ok(mean < 2.5, `mean diff vs untiled: ${mean.toFixed(2)}`);
});

test('flat color stays flat across tile seams', async () => {
  const flat = await sharp({ create: { width: 96, height: 64, channels: 3, background: { r: 128, g: 128, b: 128 } } })
    .png()
    .toBuffer();
  const { data, info } = await sharp(await upscale(flat, { tile: 48, overlap: 8, format: 'png' }))
    .raw()
    .toBuffer({ resolveWithObject: true });

  let prev;
  let maxStep = 0;
  for (let x = 0; x < info.width; x++) {
    let sum = 0;
    for (let y = 100; y < 200; y++) sum += data[(y * info.width + x) * 3];
    const avg = sum / 100;
    if (prev !== undefined) maxStep = Math.max(maxStep, Math.abs(avg - prev));
    prev = avg;
  }
  assert.ok(maxStep < 3, `column step at seam: ${maxStep}`);
});

test('rejects invalid options before inference', async () => {
  const cases = [
    [{ tile: 0 }, /tile/],
    [{ tile: 12.5 }, /tile/],
    [{ overlap: -1 }, /overlap/],
    [{ overlap: 512 }, /overlap/],
    [{ height: 0 }, /height/],
    [{ format: 'gif' }, /format/],
    [{ quality: 0 }, /quality/],
    [{ quality: 101 }, /quality/],
    [{ maxOutputPixels: 0 }, /maxOutputPixels/],
    [{ maxOutputPixels: 1.5 }, /maxOutputPixels/],
  ];
  for (const [opts, re] of cases) await assert.rejects(upscale(FIXTURE, opts), re);
  await assert.rejects(upscale(FIXTURE, null), /opts must be an object/);
});

test('enforces the output pixel cap', async () => {
  await assert.rejects(upscale(FIXTURE, { maxOutputPixels: 100_000 }), /above maxOutputPixels/);
  const tiny = await sharp({ create: { width: 16, height: 16, channels: 3, background: { r: 10, g: 90, b: 200 } } })
    .png()
    .toBuffer();
  const m = await meta(await upscale(tiny, { maxOutputPixels: Infinity, format: 'png' }));
  assert.deepEqual([m.width, m.height], [64, 64]);
});
