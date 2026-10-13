"""Does physical ink-bleed correlate with discrepancies? Full-corpus, per-line (spec §6.5.8/§6.7).

Joins the corpus-wide per-line ink-bleed score (the §6.5.8 filtering run — one consistent
normalisation over all 13,677 kept lines) with each leader's per-line discrepancy counts from
``line_diff.json``. Beyond the overall correlation, the decomposition is the point: bleed is a
PHYSICAL degradation, so it should drive the *recognition* signal (substitutions; the
probable-misread subclass) and leave the *convention* signal (spacing/abbreviation habits the
edition normalises; scribal-convention subclass) untouched — a validity check for both the bleed
metric and the discrepancy taxonomy.

    PROJECT_ROOT=. uv run python scripts/ocr/bleed_discrepancy_correlation.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

CATS = [
    "punctuation",
    "substitution",
    "spacing",
    "orthographic",
    "abbreviation",
    "deletion",
    "addition",
]
SUBCLASSES = ["scribal-convention", "attested-variant", "probable-misread", "uncertain"]
MODELS = {
    "kraken+LM": "data/processed/transcription/krakenLM_full_corpus",
    "TrOCR": "data/processed/transcription/vitlightreal_full_corpus",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bleed",
        type=Path,
        default=Path(
            "data/processed/filtered_images/20260618_160948/ink_bleed_20260618_161555.json"
        ),
    )
    a = ap.parse_args()
    bleed_raw = json.loads(a.bleed.read_text())["images"]
    bleed = {Path(k).stem: v["bleed_score"] for k, v in bleed_raw.items()}

    for model, root in MODELS.items():
        root = Path(root)
        diffs = json.loads((root / "line_diff.json").read_text())
        rows = []  # per line: bleed, chars, counts
        for page_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            page = page_dir.name
            by_line = diffs.get(page, {}).get("by_line", {})
            for f in page_dir.glob(f"{page}_line_*.txt"):
                m = re.search(r"_line_(\d+)$", f.stem)
                if not m or f.stem not in bleed:
                    continue
                dl = by_line.get(m.group(1), [])
                cat = Counter(d["type"] for d in dl)
                sub = Counter(d.get("subclass") for d in dl if d["group"] == "substantive")
                rows.append(
                    {
                        "bleed": bleed[f.stem],
                        "chars": max(1, len(f.read_text(encoding="utf-8").strip())),
                        "total": len(dl),
                        **{c: cat.get(c, 0) for c in CATS},
                        **{s: sub.get(s, 0) for s in SUBCLASSES},
                    }
                )
        b = np.array([r["bleed"] for r in rows])
        ch = np.array([r["chars"] for r in rows])
        tot = np.array([r["total"] for r in rows])
        print(f"\n=== {model} — {len(rows)} lines joined ===")
        rho_t = spearmanr(b, tot).correlation
        rho_n = spearmanr(b, 100 * tot / ch).correlation
        print(
            f"bleed ↔ total discrepancies/line : ρ = {rho_t:+.3f}   (per 100 chars: {rho_n:+.3f})"
        )
        print(f"{'category':>20} | {'ρ(bleed, count)':>15}")
        for c in CATS:
            rho = spearmanr(b, np.array([r[c] for r in rows])).correlation
            print(f"{c:>20} | {rho:>+15.3f}")
        for s in SUBCLASSES:
            rho = spearmanr(b, np.array([r[s] for r in rows])).correlation
            print(f"{('sub: ' + s):>20} | {rho:>+15.3f}")
        # quartile view (effect size in plain units)
        qs = np.quantile(b, [0.25, 0.5, 0.75])
        lab = ["Q1 (least bleed)", "Q2", "Q3", "Q4 (most bleed)"]
        idx = np.digitize(b, qs)
        print(
            f"{'bleed quartile':>18} | {'n':>5} | {'diffs/line':>10} | {'subst/line':>10} | {'misread/line':>12} | {'convention/line':>15}"
        )
        for q in range(4):
            m_ = idx == q
            subst = np.array([r["substitution"] for r in rows])[m_]
            mis = np.array([r["probable-misread"] for r in rows])[m_]
            conv = np.array([r["scribal-convention"] for r in rows])[m_]
            print(
                f"{lab[q]:>18} | {m_.sum():>5} | {tot[m_].mean():>10.2f} | {subst.mean():>10.2f} | "
                f"{mis.mean():>12.2f} | {conv.mean():>15.2f}"
            )


if __name__ == "__main__":
    main()
