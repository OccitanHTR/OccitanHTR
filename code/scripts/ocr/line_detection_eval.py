"""Held-out line-DETECTION evaluation: production blla segmentation vs hand-annotated GT.

GT: eScriptorium ALTO exports with per-line polygons
(``layout/evaluation/text_line_segments_annotations``, 6 pages never used in any training).
Predictions: the production segmentation JSONs (kraken blla + YOLO masks,
``segmentation_20260618_111517``) — i.e. the deployed pipeline's line polygons.

Protocol: polygon-mask IoU (exact, rasterised per candidate pair in the pair's union bbox),
greedy one-to-one matching by IoU, precision/recall/F1 at IoU ∈ {0.3, 0.5, 0.7}, pooled and
per page. This is the direct line-detection accuracy the alignment-based proxy (spec §4.1)
could only bound.

    PROJECT_ROOT=. uv run python scripts/ocr/line_detection_eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}


def gt_page_key(stem: str) -> str:
    """'17 - f. 012v - 013' -> '17_f_012v_013' ; '5 - garde - 001' -> '05_garde_001'."""
    parts = [re.sub(r"\s+", "_", p.strip().replace(".", "")) for p in stem.split(" - ")]
    parts[0] = parts[0].zfill(2)
    return "_".join(parts)


def parse_gt(xml_path: Path) -> list[list[tuple[int, int]]]:
    root = ET.parse(xml_path).getroot()
    polys = []
    for tl in root.findall(".//alto:TextLine", NS):
        pg = tl.find("./alto:Shape/alto:Polygon", NS)
        if pg is None:
            continue
        pts = [int(x) for x in re.split(r"[ ,]+", pg.get("POINTS").strip())]
        polys.append(list(zip(pts[0::2], pts[1::2])))
    return polys


def bbox(poly):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def pair_iou(pa, pb) -> float:
    ax0, ay0, ax1, ay1 = bbox(pa)
    bx0, by0, bx1, by1 = bbox(pb)
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
        return 0.0
    ux0, uy0 = min(ax0, bx0), min(ay0, by0)
    ux1, uy1 = max(ax1, bx1), max(ay1, by1)
    w, h = ux1 - ux0 + 1, uy1 - uy0 + 1
    ma = Image.new("1", (w, h))
    ImageDraw.Draw(ma).polygon([(x - ux0, y - uy0) for x, y in pa], fill=1)
    mb = Image.new("1", (w, h))
    ImageDraw.Draw(mb).polygon([(x - ux0, y - uy0) for x, y in pb], fill=1)
    a, b = np.array(ma, bool), np.array(mb, bool)
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union else 0.0


def match(gt, pred):
    """Greedy one-to-one matching by IoU desc; returns list of matched IoUs."""
    cands = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            iou = pair_iou(g, p)
            if iou > 0.05:
                cands.append((iou, i, j))
    cands.sort(reverse=True)
    used_g, used_p, matched = set(), set(), []
    for iou, i, j in cands:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matched.append(iou)
    return matched


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gt-dir",
        type=Path,
        default=Path(
            "data/processed/annotated_samples/layout/evaluation/text_line_segments_annotations"
        ),
    )
    ap.add_argument(
        "--seg-dir",
        type=Path,
        default=Path("data/processed/segmented_images/segmentation_20260618_111517"),
    )
    ap.add_argument(
        "--out", type=Path, default=Path("tests/ocr/evaluations/line_detection_heldout_20260913")
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    thresholds = (0.3, 0.5, 0.7)
    rows, pooled = [], {t: [0, 0, 0] for t in thresholds}  # t -> [tp, n_gt, n_pred]
    for xml in sorted(a.gt_dir.glob("*.xml")):
        if xml.name == "METS.xml":
            continue
        key = gt_page_key(xml.stem)
        segf = a.seg_dir / f"{key}.json"
        gt = parse_gt(xml)
        pred = [
            ln["boundary"] for ln in json.loads(segf.read_text())["lines"] if ln.get("boundary")
        ]
        ious = match(gt, pred)
        row = {"page": key, "n_gt": len(gt), "n_pred": len(pred)}
        for t in thresholds:
            tp = sum(1 for x in ious if x >= t)
            row[f"tp@{t}"] = tp
            pooled[t][0] += tp
            pooled[t][1] += len(gt)
            pooled[t][2] += len(pred)
        rows.append(row)
        print(
            f"{key}: GT {len(gt)} | pred {len(pred)} | "
            + " | ".join(f"TP@{t} {row[f'tp@{t}']}" for t in thresholds)
        )

    print("\n=== POOLED (held-out line detection, production blla+masks) ===")
    for t in thresholds:
        tp, ngt, npred = pooled[t]
        p = tp / npred if npred else 0
        r = tp / ngt if ngt else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        print(
            f"IoU ≥ {t}: P {p:.4f} | R {r:.4f} | F1 {f1:.4f}   (TP {tp} / GT {ngt} / pred {npred})"
        )

    with open(a.out / "per_page.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-page CSV -> {a.out}/per_page.csv")


if __name__ == "__main__":
    main()
