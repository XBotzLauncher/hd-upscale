# hd-upscale

Real-ESRGAN 4x upscaler for Node.js. Runs on CPU through ONNX Runtime — no Python, no GPU.

Model: `realesr-general-x4v3` (Real-ESRGAN, BSD-3-Clause), bundled in `models/` (~4.7 MB).

## Usage

```bash
npm install                    # onnxruntime-node + sharp
node cli.js photo.jpg          # -> photo_4x.jpg
npx hd-upscale photo.jpg       # same, once the package is installed/published
```

## API

```js
import { upscale, upscaleFile } from 'hd-upscale';

// Buffer / path -> Buffer
const jpeg = await upscale('photo.jpg', {
  height: 1920,        // optional: downscale the 4x result to this height (sharper & smaller)
  format: 'jpeg',      // jpeg | png | webp, default jpeg
  quality: 95,
  tile: 512,           // inference tile size; lower it if RAM is limited
  overlap: 32,         // overlap between tiles, cross-faded (feathered) at seams
  maxOutputPixels: 64_000_000, // cap on the 4x output (w*h*16); Infinity disables
});

// Write directly to disk and get metadata
const info = await upscaleFile('photo.jpg', 'photo_hd.jpg', { height: 1920 });
console.log(info.width, info.height);
```

Input can be a `Buffer` (upload) or a file path. EXIF orientation is honored, and PNG alpha is preserved.

All options are validated before any inference: `tile` (positive integer), `overlap` (0 ≤ overlap < tile), `height` (positive integer), `format`, `quality` (1–100), `maxOutputPixels` (positive integer or `Infinity`). Invalid values throw `RangeError`/`TypeError` immediately.

## Output pixel limit

The default cap is **64 MP** for the 4x result (input up to 4 MP). It is checked from the image header before pixels are decoded, so oversized uploads fail fast without allocating memory. `height` does not bypass the cap because inference still runs at 4x; raise or disable it explicitly if your server can handle it:

```js
await upscale(buf, { maxOutputPixels: 192_000_000 }); // 12 MP in -> 192 MP out
await upscale(buf, { maxOutputPixels: Infinity });
```

## TypeScript

Types are bundled (`src/index.d.ts`, `upscale`/`upscaleFile`/`UpscaleOptions`); no `@types` package needed.

## Express integration

```js
import express from 'express';
import multer from 'multer';
import { upscale } from 'hd-upscale';

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 15 << 20 } });

app.post('/api/upscale', upload.single('image'), async (req, res) => {
  try {
    const out = await upscale(req.file.buffer, { height: 1920, format: 'jpeg' });
    res.type('image/jpeg').send(out);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});
```

Frontend:

```js
const fd = new FormData();
fd.append('image', file);
const res = await fetch('/api/upscale', { method: 'POST', body: fd });
const blob = await res.blob();
```

## Production notes

- **Speed**: ~40 seconds for a 415x739 image on an 8-core CPU. Lower `tile` if RAM is tight (memory is roughly `tile² × 64 × 4 bytes` per activation).
- **Seams**: tile overlaps are cross-faded with a linear alpha ramp, so boundary pixels blend between neighbouring tile predictions instead of being hard-pasted. `overlap: 32` is enough for the compact model; raising it costs time roughly linearly.
- **Queueing**: one inference uses all cores. Limit concurrent requests to 1–2 (queue) so the server does not run out of RAM/CPU.
- **Output size**: a 4x result is 16x the input pixels. A 12 MP photo becomes 192 MP; use `height` to cap it, e.g. 1920 or 2560.
- **Input limit**: sharp by default rejects images larger than ~268 MP; set `sharp.cache(false)` / `limitInputPixels` if needed.
- **GPU discovery warning** from onnxruntime-node (`/sys/class/drm`) can be safely ignored — it only reports that no GPU is available.
- The model is fixed at 4x; `height` downscales the 4x result with Lanczos (supersampling), which is sharper than upscaling directly to that height.

## License

- **Code** (`src/`, `cli.js`, `test.js`): MIT — see [LICENSE](LICENSE).
- **Model weights** (`models/realesrgan-x4v3.onnx`): BSD 3-Clause, © 2021 Xintao Wang (Real-ESRGAN). Full text in [models/LICENSE](models/LICENSE), attribution in [NOTICE](NOTICE).

When publishing a fork, ship `LICENSE`, `NOTICE`, and `models/LICENSE`. `package.json` declares `"license": "MIT"`; switch it to `"BSD-3-Clause"` if the whole package should follow the model license.

## Tests

```bash
npm test          # node --test, uses test/fixtures/sample.png
npm run typecheck # tsc on the declaration file
npm run bench     # timings for default vs smaller tiles
```

CI runs typecheck and tests on Node 22 and 24 (`.github/workflows/ci.yml`).

## Structure

```
src/index.js          library
src/index.d.ts        TypeScript types
cli.js                CLI example
bench.js              npm run bench
test/upscale.test.js  node:test suite
test/fixtures/        real photo fixture (+ source/license note)
models/               realesrgan-x4v3.onnx (+ model LICENSE)
LICENSE               MIT (code)
NOTICE                Real-ESRGAN attribution
```
