import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import os from 'node:os';
import sharp from 'sharp';
import { upscale } from './src/index.js';

const input = await readFile(fileURLToPath(new URL('./test/fixtures/sample.png', import.meta.url)));
const { width, height } = await sharp(input).metadata();
const mp = ((width * 4 * height * 4) / 1e6).toFixed(2);

const cases = [
  { name: 'default (tile 512)', opts: {}, runs: 3 },
  { name: 'tile 128 / overlap 16', opts: { tile: 128, overlap: 16 }, runs: 2 },
  { name: 'tile 64 / overlap 8', opts: { tile: 64, overlap: 8 }, runs: 2 },
];

console.log(`hd-upscale — Node ${process.version}, ${os.cpus().length} cores, ${os.platform()}/${os.arch()}`);
console.log(`input ${width}x${height} -> ${width * 4}x${height * 4} (${mp} MP output)\n`);

const rss0 = process.memoryUsage().rss;
const rows = [];
let cold = null;

for (const { name, opts, runs } of cases) {
  const times = [];
  for (let i = 0; i < runs; i++) {
    const t = performance.now();
    await upscale(input, { ...opts, format: 'jpeg' });
    times.push(performance.now() - t);
  }
  const best = Math.min(...times);
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  if (cold === null) cold = times[0];
  rows.push({
    case: name,
    'cold ms': name.startsWith('default') ? times[0].toFixed(0) : '',
    'best ms': best.toFixed(0),
    'avg ms': avg.toFixed(0),
    'MP/s': (Number(mp) / (best / 1000)).toFixed(2),
    'rss +MB': ((process.memoryUsage().rss - rss0) / 1e6).toFixed(0),
  });
}

console.table(rows);
console.log(`\ncold start includes ONNX session + model load: ${cold.toFixed(0)} ms`);
console.log('numbers vary by CPU; one inference uses all cores');
