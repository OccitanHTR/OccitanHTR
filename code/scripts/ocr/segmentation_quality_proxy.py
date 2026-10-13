"""Line-DETECTION quality proxy against the scholarly edition (no new annotation).

The 300-val OCR metrics measure recognition CONDITIONAL on segmentation: a line the segmenter
missed or merged never became a crop, so detection quality is a separate, so-far unmeasured axis.
This uses the scholarly edition as an independent witness of how many text lines each page truly
has, via the §6.6 content-based alignment of the deployed (kraken+LM) full-corpus transcription:

  * detection RECALL proxy   — share of scholarly lines with a confidently-aligned detected line
    (an unmatched scholarly line = likely missed/merged detection OR unalignably-bad recognition;
    the proxy is therefore a LOWER bound on detection recall);
  * detection PRECISION proxy — share of detected lines aligned to some scholarly line (unmatched
    detected lines include LEGITIMATE non-edition matter — marginalia, catchwords, decorations —
    so this too is a lower bound);
  * per-page line-count agreement (detected − scholarly), split (2+ detected lines on one
    scholarly line) and merge (one detected line spanning 2+ scholarly lines) indicators.

    PROJECT_ROOT=. uv run python scripts/ocr/segmentation_quality_proxy.py \
        --transcription-dir data/processed/transcription/krakenLM_full_corpus \
        --scholarly tests/ocr/AlbucE_aligned_20260628_142959.txt \
        --out tests/ocr/evaluations/seg_quality_proxy_20260911
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def scholarly_counts(path: Path) -> dict[str, int]:
    """page_key -> number of non-empty scholarly lines (headers '== IMAGE: <page>_full ==')."""
    counts: dict[str, int] = {}
    page = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"=+ IMAGE: (.+?)_full =+", raw.strip())
        if m:
            page = m.group(1)
            counts[page] = 0
            continue
        if page and raw.strip():
            counts[page] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--transcription-dir",
        type=Path,
        default=Path("data/processed/transcription/krakenLM_full_corpus"),
    )
    ap.add_argument(
        "--scholarly", type=Path, default=Path("tests/ocr/AlbucE_aligned_20260628_142959.txt")
    )
    ap.add_argument(
        "--score-thr",
        type=float,
        default=0.5,
        help="alignment score for a 'confident' match (sensitivity also at 0.7)",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("tests/ocr/evaluations/seg_quality_proxy_20260911")
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    align = json.loads((a.transcription_dir / "line_alignment.json").read_text(encoding="utf-8"))
    sch = scholarly_counts(a.scholarly)

    rows = []
    T = Counter()
    for page in sorted(align.keys()):
        pairs = align[page].get("pairs", [])
        n_det = len(list((a.transcription_dir / page).glob(f"{page}_line_*.txt")))
        n_sch = sch.get(page, 0)
        for thr_name, thr in (("t05", a.score_thr), ("t07", 0.7)):
            good = [p for p in pairs if p.get("score", 0) >= thr]
            m_sch = {p["scholarly_no"] for p in good}
            m_det = {p["model_idx"] for p in good}
            splits = sum(1 for _, c in Counter(p["scholarly_no"] for p in good).items() if c >= 2)
            merges = sum(1 for _, c in Counter(p["model_idx"] for p in good).items() if c >= 2)
            T[f"{thr_name}_msch"] += len(m_sch)
            T[f"{thr_name}_mdet"] += len(m_det)
            T[f"{thr_name}_splits"] += splits
            T[f"{thr_name}_merges"] += merges
            if thr_name == "t05":
                rows.append(
                    {
                        "page": page,
                        "n_detected": n_det,
                        "n_scholarly": n_sch,
                        "matched_scholarly": len(m_sch),
                        "matched_detected": len(m_det),
                        "recall_proxy": round(len(m_sch) / n_sch, 4) if n_sch else "",
                        "precision_proxy": round(len(m_det) / n_det, 4) if n_det else "",
                        "count_delta": n_det - n_sch,
                        "splits": splits,
                        "merges": merges,
                    }
                )
        T["det"] += n_det
        T["sch"] += n_sch

    with open(a.out / "per_page.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_pages = len(rows)
    print(
        f"{n_pages} pages | detected lines {T['det']} | scholarly lines {T['sch']} "
        f"(count ratio {T['det'] / T['sch']:.3f})"
    )
    for nm, thr in (("t05", a.score_thr), ("t07", 0.7)):
        print(
            f"score ≥ {thr}: recall proxy {T[f'{nm}_msch'] / T['sch']:.3f} "
            f"({T[f'{nm}_msch']}/{T['sch']} scholarly lines matched) | "
            f"precision proxy {T[f'{nm}_mdet'] / T['det']:.3f} "
            f"({T[f'{nm}_mdet']}/{T['det']} detected lines matched) | "
            f"splits {T[f'{nm}_splits']} | merges {T[f'{nm}_merges']}"
        )
    worst = sorted((r for r in rows if r["recall_proxy"] != ""), key=lambda r: r["recall_proxy"])[
        :8
    ]
    print(f"\nworst pages by recall proxy (score ≥ {a.score_thr:.1f}):")
    for r in worst:
        print(
            f"  {r['page']:<18} recall {r['recall_proxy']:<7} precision {r['precision_proxy']:<7} "
            f"det {r['n_detected']:>3} sch {r['n_scholarly']:>3} Δ {r['count_delta']:+d}"
        )
    print(f"\nper-page CSV -> {a.out}/per_page.csv")


if __name__ == "__main__":
    main()
