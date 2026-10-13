"""Discrepancy funnel + corpus descriptives + per-page plots (spec §6.7).

From a model's full-corpus ``line_diff.json`` + its per-line transcription files:
  1. FUNNEL — total lines → lines without any discrepancy → for lines with ≥1: mean/median
     discrepancies per line, plus length-normalised rates (per 100 characters) so long lines
     don't masquerade as hard ones.
  2. DESCRIPTIVES — pages, lines, lines-per-page and characters-per-line distributions.
  3. PLOTS — (a) bar chart of total discrepancies per page (style of error_distribution.png);
     (b) line plot of per-page counts split by category, one colour per category.

    PROJECT_ROOT=. uv run python scripts/ocr/discrepancy_page_stats.py \
        --transcription-dir data/processed/transcription/krakenLM_full_corpus \
        --label "kraken+LM (deployed)" \
        --out tests/ocr/evaluations/discrepancy_page_plots_20260912
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CATS = [
    "punctuation",
    "substitution",
    "spacing",
    "orthographic",
    "abbreviation",
    "deletion",
    "addition",
]
# Material-300 pastels, matching the error_distribution.png house style
COLORS = {
    "punctuation": "#90a4ae",
    "substitution": "#e57373",
    "spacing": "#ffb74d",
    "orthographic": "#81c784",
    "abbreviation": "#64b5f6",
    "deletion": "#ba68c8",
    "addition": "#a1887f",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--transcription-dir",
        type=Path,
        default=Path("data/processed/transcription/krakenLM_full_corpus"),
    )
    ap.add_argument("--label", default="kraken+LM (deployed)")
    ap.add_argument(
        "--out", type=Path, default=Path("tests/ocr/evaluations/discrepancy_page_plots_20260912")
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    diffs = json.loads((a.transcription_dir / "line_diff.json").read_text(encoding="utf-8"))

    # per-line inventory from the transcription files (line id -> char count)
    chars: dict[tuple[str, str], int] = {}
    for page_dir in sorted(p for p in a.transcription_dir.iterdir() if p.is_dir()):
        page = page_dir.name
        for f in page_dir.glob(f"{page}_line_*.txt"):
            m = re.search(r"_line_(\d+)$", f.stem)
            if not m:  # skip any auxiliary files that aren't plain per-line outputs
                continue
            chars[(page, m.group(1))] = len(f.read_text(encoding="utf-8").strip())

    # per-line diff counts (+ per-page per-category)
    ndiff: dict[tuple[str, str], int] = Counter()
    page_cat: dict[str, Counter] = defaultdict(Counter)
    for page, pv in diffs.items():
        for idx, dl in pv.get("by_line", {}).items():
            ndiff[(page, idx)] += len(dl)
            for df in dl:
                page_cat[page][df["type"]] += 1

    # ---- 1. funnel ----
    total = len(chars)
    with_d = [k for k in chars if ndiff.get(k, 0) > 0]
    clean = total - len(with_d)
    per_line = np.array([ndiff[k] for k in with_d])
    lens = np.array([max(1, chars[k]) for k in with_d])
    per100 = 100.0 * per_line / lens
    all_chars = sum(chars.values())
    all_diffs = int(per_line.sum())
    print(f"=== FUNNEL — {a.label} ===")
    print(f"total lines               : {total}  ({len(diffs)} pages)")
    print(f"lines with NO discrepancy : {clean}  ({100 * clean / total:.1f}%)")
    print(f"lines with ≥1             : {len(with_d)}  ({100 * len(with_d) / total:.1f}%)")
    print(
        f"  discrepancies/line      : mean {per_line.mean():.2f} | median {np.median(per_line):.0f} "
        f"| p90 {np.percentile(per_line, 90):.0f} | max {per_line.max()}"
    )
    print(
        f"  length-normalised       : mean {per100.mean():.2f} per 100 chars "
        f"(median {np.median(per100):.2f}); corpus rate {100 * all_diffs / all_chars:.2f}/100 chars "
        f"({all_diffs} diffs / {all_chars} chars)"
    )

    # ---- 2. descriptives ----
    lp = Counter(p for p, _ in chars)
    lpp = np.array([lp[p] for p in sorted(lp)])
    cl = np.array([c for c in chars.values()])

    def stats(x):
        return (
            f"mean {x.mean():.1f} | sd {x.std():.1f} | min {x.min()} | p25 {np.percentile(x, 25):.0f} "
            f"| median {np.median(x):.0f} | p75 {np.percentile(x, 75):.0f} | max {x.max()}"
        )

    print("\n=== DESCRIPTIVES ===")
    print(f"pages: {len(lp)} | lines: {total}")
    print(f"lines/page : {stats(lpp)}")
    print(f"chars/line : {stats(cl)}")

    # ---- 3a. bar: discrepancies per page ----
    pages = sorted(lp)
    tot_page = [sum(page_cat[p].values()) for p in pages]
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(len(pages)), tot_page, color="#e57373")
    ax.axhline(np.mean(tot_page), ls="--", c="k", lw=1, label=f"mean {np.mean(tot_page):.0f}/page")
    ax.set_xticks(range(len(pages)))
    ax.set_xticklabels(pages, rotation=90, fontsize=5)
    ax.set(ylabel="discrepancies", title=f"Discrepancies per page — {a.label}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(a.out / "discrepancies_per_page.png", dpi=150)

    # ---- 3b. line plot per category ----
    fig, ax = plt.subplots(figsize=(16, 5.5))
    for c in CATS:
        ax.plot(
            range(len(pages)),
            [page_cat[p].get(c, 0) for p in pages],
            lw=1.4,
            marker=".",
            ms=4,
            color=COLORS[c],
            label=c,
        )
    ax.set_xticks(range(len(pages)))
    ax.set_xticklabels(pages, rotation=90, fontsize=5)
    ax.set(ylabel="discrepancies", title=f"Discrepancies per page by category — {a.label}")
    ax.legend(frameon=False, ncol=4, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(a.out / "discrepancies_per_page_by_category.png", dpi=150)

    # dump per-page CSV for reproducibility
    with open(a.out / "per_page_categories.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page", "n_lines", "total", *CATS])
        for p in pages:
            w.writerow(
                [p, lp[p], sum(page_cat[p].values()), *[page_cat[p].get(c, 0) for c in CATS]]
            )
    print(
        f"\nsaved {a.out}/discrepancies_per_page.png, discrepancies_per_page_by_category.png, per_page_categories.csv"
    )


if __name__ == "__main__":
    main()
