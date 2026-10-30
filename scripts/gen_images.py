#!/usr/bin/env python3
"""Write synthetic RGB test images at the benchmark sizes into data/ (gitignored).

Usage: python scripts/gen_images.py [--sizes 512x512 1024x1024 ...] [--out data] [--seed 1]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_SIZES = ["512x512", "1024x1024", "2048x2048", "4096x4096", "7680x4320", "16384x16384"]


def make_image(w: int, h: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    r = (255 * x / max(w - 1, 1)).astype(np.int32)
    g = (255 * y / max(h - 1, 1)).astype(np.int32)
    cell = min(w, h) // 8 + 1
    b = np.where(((x // cell) + (y // cell)) % 2 == 0, 220, 128)
    disc = (x - w // 2) ** 2 + (y - h // 2) ** 2 < (min(w, h) // 3) ** 2
    r = np.where(disc, 255 - r, r)
    g = np.where(disc, 255 - g, g)
    noise = rng.integers(-16, 17, size=(h, w))
    img = np.stack([r + noise, g + noise, b + noise], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None
    for s in args.sizes:
        w, h = (int(v) for v in s.lower().split("x"))
        path = out / f"synth_{w}x{h}.png"
        Image.fromarray(make_image(w, h, args.seed), "RGB").save(path)
        print(path)


if __name__ == "__main__":
    main()
