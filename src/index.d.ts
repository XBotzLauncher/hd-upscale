import type { Metadata } from 'sharp';

export interface UpscaleOptions {
  /** Inference tile size in source pixels. Default 512. Must be a positive integer. */
  tile?: number;
  /** Tile overlap, cross-faded at seams. Default 32. Must be an integer >= 0 and < tile. */
  overlap?: number;
  /** Downscale the 4x result to this height (Lanczos). */
  height?: number;
  /** Output encoding. Default 'jpeg'. */
  format?: 'jpeg' | 'png' | 'webp';
  /** Encoder quality for jpeg/webp, 1-100. Default 95. */
  quality?: number;
  /** Cap on the 4x output pixels (width * height * 16). Default 64_000_000; Infinity disables. */
  maxOutputPixels?: number;
}

/** Default cap: 64 MP for the 4x result. */
export declare const DEFAULT_MAX_OUTPUT_PIXELS: number;

/**
 * Upscale an image 4x with Real-ESRGAN on CPU.
 *
 * @param input image buffer or file path
 * @throws {RangeError|TypeError} invalid options or output above maxOutputPixels
 */
export declare function upscale(
  input: Buffer | string | Uint8Array,
  opts?: UpscaleOptions,
): Promise<Buffer>;

/**
 * Upscale a file and write the result to disk.
 *
 * @returns metadata of the written image
 */
export declare function upscaleFile(
  src: string,
  dst: string,
  opts?: UpscaleOptions,
): Promise<Metadata>;
