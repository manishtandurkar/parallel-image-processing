#!/usr/bin/env python3
"""Write the sample images for the report and slides into report/figs/samples/.

Runs pagp_seq on a synthetic input and saves input, gray, blur_sep, sobel, pipeline, plus a
labelled side-by-side montage. Every other backend matches these outputs (max_abs_diff 0 in all
result CSVs), so the seq images stand for all of them.

Usage: python scripts/sample_images.py [--size 512x512] [--build build] [--out report/figs/samples]
"""
import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from gen_images import make_image

OPS = ["gray", "blur_sep", "sobel", "pipeline"]
LABELS = {
    "input": "Input (RGB)",
    "gray": "Grayscale",
    "blur_sep": "Gaussian blur 5x5",
    "sobel": "Sobel (no blur)",
    "pipeline": "Pipeline: gray > blur > Sobel",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="512x512")
    ap.add_argument("--build", default="build")
    ap.add_argument("--out", default="report/figs/samples")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    src = out / "input.png"
    Image.fromarray(make_image(w, h, seed=1), "RGB").save(src)
    for op in OPS:
        subprocess.run([f"{args.build}/pagp_seq", "--input", str(src), "--op", op, "--reps", "1",
                        "--save", str(out / f"{op}.png")], check=True, stdout=subprocess.DEVNULL)

    names = ["input"] + OPS
    tile, pad, text_h = 256, 8, 22
    montage = Image.new("RGB", (len(names) * (tile + pad) + pad, tile + text_h + 2 * pad), "white")
    draw = ImageDraw.Draw(montage)
    for i, name in enumerate(names):
        im = Image.open(out / f"{name}.png").convert("RGB").resize((tile, tile), Image.LANCZOS)
        x = pad + i * (tile + pad)
        montage.paste(im, (x, pad))
        draw.text((x, pad + tile + 4), LABELS[name], fill="black")
    montage.save(out / "montage.png")
    for p in sorted(out.glob("*.png")):
        print(p, f"{p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
