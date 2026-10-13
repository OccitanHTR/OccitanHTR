"""Page x vertical-zone heatmap of CER (companion to error_distribution.py).

The per-page error bar chart hides WHERE on the page the errors sit. This heatmap keeps
pages on the x-axis and divides each page's full line sequence — numeration (reading)
order — into N bands (default 4): band 1 = top, band N = bottom. Each annotated GT line
is placed in a band by its line index relative to the page's total segmented lines, and
the cell shows the mean per-line CER (clipped) of the GT lines in that zone. Cells with
no annotated line are grey.

Uses the stored predictions of a model that covers the whole corpus (default: frozen
catmus, unbiased — never trained on any manuscript line), so no inference is needed.

    PROJECT_ROOT=. uv run python scripts/ocr/error_zone_heatmap.py
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import matplotlib
import numpy as np
from rapidfuzz.distance import Levenshtein

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("data/processed/transcription/ocr_kept_20260622_120413"),
    )
    ap.add_argument(
        "--gt-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("data/processed/annotated_samples/OCR/full_annotated"),
            Path("data/processed/annotated_samples/OCR/validation"),
        ],
    )
    ap.add_argument("--label", default="frozen catmus, 899 annotated lines")
    ap.add_argument("--bands", type=int, default=4)
    ap.add_argument(
        "--out", type=Path, default=Path("tests/ocr/evaluations/error_distribution_catmus")
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    # total segmented lines per page (for relative vertical position)
    page_total: dict[str, int] = {}
    for page_dir in sorted(p for p in a.pred_dir.iterdir() if p.is_dir()):
        n = sum(1 for f in page_dir.glob(f"{page_dir.name}_line_*.txt"))
        if n:
            page_total[page_dir.name] = n

    # per-GT-line CER, banded by relative line position
    cells: dict[tuple[int, str], list[float]] = {}
    n_lines = 0
    for gd in a.gt_dirs:
        for f in sorted(gd.glob("*.gt.txt")):
            gt = nfc(f.read_text(encoding="utf-8"))
            if not gt:
                continue
            stem = f.name.removesuffix(".gt.txt")
            m = re.match(r"(.+)_line_(\d+)$", stem)
            page, idx = m.group(1), int(m.group(2))
            pred_f = a.pred_dir / page / f"{stem}.txt"
            if page not in page_total or not pred_f.exists():
                continue
            pred = nfc(pred_f.read_text(encoding="utf-8"))
            cer = min(Levenshtein.distance(gt, pred), len(gt)) / len(gt)
            band = min(a.bands - 1, int(a.bands * idx / page_total[page]))
            cells.setdefault((band, page), []).append(cer)
            n_lines += 1

    pages = sorted(page_total)
    H = np.full((a.bands, len(pages)), np.nan)
    for (b, page), cers in cells.items():
        H[b, pages.index(page)] = float(np.mean(cers))

    fig, ax = plt.subplots(figsize=(16, 3.6))
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("#eceff1")  # no annotated line in this zone
    im = ax.imshow(H, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(pages)))
    ax.set_xticklabels(pages, rotation=90, fontsize=5)
    ax.set_yticks(range(a.bands))
    ax.set_yticklabels(
        [
            f"band {b + 1}" + (" (top)" if b == 0 else " (bottom)" if b == a.bands - 1 else "")
            for b in range(a.bands)
        ],
        fontsize=8,
    )
    ax.set(title=f"Mean per-line CER by page and vertical zone — {a.label}")
    fig.colorbar(im, ax=ax, pad=0.01, label="mean per-line CER")
    fig.tight_layout()
    out = a.out / f"error_zone_heatmap_{a.bands}bands.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}  ({n_lines} GT lines, {len(pages)} pages, {a.bands} bands)")

    band_mean = np.nanmean(H, axis=1)
    print(f"corpus mean per-line CER per band (1=top .. {a.bands}=bottom):")
    for b, v in enumerate(band_mean):
        n_cells = int(np.sum(~np.isnan(H[b])))
        print(f"  band {b + 1}: {v:.4f}  ({n_cells} pages with data)")


if __name__ == "__main__":
    main()
