"""Build the multi-font figure: real crop + ONLY the 6 fonts of the MF pool (spec §6.5.19).

Extracts the relevant rows from the 13-font sample sheets
(``tests/ocr/font_samples_20260730/fonts_vs_real__*.png``): the real manuscript crop,
the single-font baseline (merged_font_code_cmpl2), and the five curated Gothic faces
(Missaali, Jena1330, _aeiou2U, xenipp3U, xibern2U). No highlighting.

    PROJECT_ROOT=. uv run python scripts/ocr/mf_font_figure.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

SRC_DIR = Path("tests/ocr/font_samples_20260730")
N_ROWS = 14  # rows in the source sheets, uniformly spaced
# source row index -> label (order of the original sheet)
KEEP = [
    0,
    10,
    6,
    5,
    8,
    12,
    13,
]  # REAL, merged(our font), Missaali, Jena1330, _aeiou2U, xenipp3U, xibern2U


def main() -> None:
    for src in sorted(SRC_DIR.glob("fonts_vs_real__*.png")):
        if "_mf6" in src.stem:
            continue
        im = Image.open(src).convert("RGB")
        w, h = im.size
        bands = []
        for i in KEEP:
            y0, y1 = round(i * h / N_ROWS), round((i + 1) * h / N_ROWS)
            bands.append(im.crop((0, y0, w, y1)))
        out_h = sum(b.height for b in bands)
        canvas = Image.new("RGB", (w, out_h), "#ffffff")
        y = 0
        for b in bands:
            canvas.paste(b, (0, y))
            y += b.height
        out = src.with_name(src.stem.replace("fonts_vs_real__", "fonts_vs_real_mf6__") + ".png")
        canvas.save(out)
        print(f"saved {out}  ({w}x{out_h})")


if __name__ == "__main__":
    main()
