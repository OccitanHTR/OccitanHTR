"""Page x vertical-zone heatmap of discrepancies (companion to discrepancy_page_stats.py).

The per-page bar chart hides WHERE on the page the discrepancies sit. This heatmap keeps
pages on the x-axis and divides each page's lines — in numeration (reading) order — into
N equal bands (default 6): band 1 = top of the page, band N = bottom. Cell value = mean
discrepancies per line in that band, so bands of different sizes are comparable and a
dark cell reads as "this zone of this page is problematic".

    PROJECT_ROOT=. uv run python scripts/ocr/discrepancy_zone_heatmap.py \
        --transcription-dir data/processed/transcription/krakenLM_full_corpus \
        --label "kraken+LM (deployed)" \
        --out tests/ocr/evaluations/discrepancy_page_plots_20260912/krakenLM
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--transcription-dir",
        type=Path,
        default=Path("data/processed/transcription/krakenLM_full_corpus"),
    )
    ap.add_argument("--label", default="kraken+LM (deployed)")
    ap.add_argument("--bands", type=int, default=6)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("tests/ocr/evaluations/discrepancy_page_plots_20260912/krakenLM"),
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    diffs = json.loads((a.transcription_dir / "line_diff.json").read_text(encoding="utf-8"))

    # line inventory per page (numeric order) from the transcription files
    page_lines: dict[str, list[int]] = {}
    for page_dir in sorted(p for p in a.transcription_dir.iterdir() if p.is_dir()):
        idxs = []
        for f in page_dir.glob(f"{page_dir.name}_line_*.txt"):
            m = re.search(r"_line_(\d+)$", f.stem)
            if m:
                idxs.append(int(m.group(1)))
        if idxs:
            page_lines[page_dir.name] = sorted(idxs)

    ndiff: dict[str, Counter] = {}
    for page, pv in diffs.items():
        ndiff[page] = Counter({idx: len(dl) for idx, dl in pv.get("by_line", {}).items()})

    pages = sorted(page_lines)
    H = np.zeros((a.bands, len(pages)))
    for j, page in enumerate(pages):
        idxs = page_lines[page]
        splits = np.array_split(np.array(idxs), a.bands)  # numeration order -> N bands
        for b, band in enumerate(splits):
            if len(band) == 0:
                H[b, j] = np.nan
                continue
            H[b, j] = np.mean([ndiff.get(page, {}).get(str(i), 0) for i in band])

    fig, ax = plt.subplots(figsize=(16, 4.2))
    im = ax.imshow(H, aspect="auto", cmap="Reds", interpolation="nearest")
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
    ax.set(title=f"Discrepancies per line by page and vertical zone — {a.label}")
    fig.colorbar(im, ax=ax, pad=0.01, label="discrepancies / line")
    fig.tight_layout()
    out = a.out / f"zone_heatmap_{a.bands}bands.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    band_mean = np.nanmean(H, axis=1)
    print(f"corpus mean discrepancies/line per band (1=top .. {a.bands}=bottom):")
    for b, v in enumerate(band_mean):
        print(f"  band {b + 1}: {v:.2f}")


if __name__ == "__main__":
    main()
