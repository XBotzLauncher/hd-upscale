#!/usr/bin/env node
// Example: node cli.js input.jpg [output.jpg] [--height 1920] [--format jpeg]
import { upscaleFile } from './src/index.js';

const args = process.argv.slice(2);
const flag = (name) => {
  const i = args.indexOf(`--${name}`);
  return i === -1 ? undefined : args[i + 1];
};
const files = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')) && !/^\d+$/.test(a));
const [src, dst = src?.replace(/(\.\w+)?$/, '_4x$1')] = files;

if (!src) {
  console.error('usage: node cli.js <input> [output] [--height N] [--format jpeg|png|webp]');
  process.exit(1);
}

const format = flag('format') ?? dst.match(/\.(png|webp|jpe?g)$/i)?.[1].toLowerCase().replace('jpg', 'jpeg');
const t = Date.now();
const info = await upscaleFile(src, dst, {
  height: flag('height') ? Number(flag('height')) : undefined,
  format,
});
console.log(`${dst} ${info.width}x${info.height} (${((Date.now() - t) / 1000).toFixed(1)}s)`);
