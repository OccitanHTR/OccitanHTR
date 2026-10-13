"""Ground-truth validation of the substantive-discrepancy subclassifier (spec §6.7).

The subclassifier labels model↔edition discrepancies without seeing the manuscript truth.
On the 300-line held-out validation set we DO have the truth (human GT), so for every
substantive discrepancy on a val-300 line classified *probable-misread* or *uncertain* we can
adjudicate it: align the model's full-corpus line to the GT (character-level), extract the GT
segment at the discrepancy's locus, and check which side the GT supports.

A discrepancy IS a misreading iff the model's reading deviates from the GT at the locus
(what the manuscript actually says), letters and marks — spacing excluded, since spacing is
its own taxonomy category and only leaks in here via multi-word substitution spans.

Verdicts per discrepancy:
  not-misread (GT=model)    — GT equals the model reading exactly: the model faithfully read
                              the page; the discrepancy is edition-side normalisation/emendation.
  not-misread (spacing-only)— GT equals the model reading up to spaces: all letters correct.
  misread (GT=edition)      — GT equals the edition reading: misread, fully confirmed.
  misread (marks-only)      — model letters correct but combining marks dropped/added
                              (lost abbreviation mark): a misread of the mark.
  misread (other)           — model letters deviate from GT (GT matches neither side exactly;
                              typically a small misread plus edition-side normalisation).
  unlocatable               — the diff's ocr_text could not be located in the stored model line.

    PROJECT_ROOT=. uv run python scripts/ocr/validate_subclass_val300.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

MODELS = {
    "kraken+LM": "data/processed/transcription/krakenLM_full_corpus",
    "TrOCR": "data/processed/transcription/vitlightreal_full_corpus",
}
SUBCLASSES = ("probable-misread", "uncertain")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def skeleton(s: str) -> str:
    """Spaces removed, combining marks stripped (NFD) — the bare letter sequence."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.replace(" ", "")) if not unicodedata.combining(c)
    )


def map_span(model_line: str, gt: str, i0: int, i1: int) -> str:
    """GT segment corresponding to model_line[i0:i1] via char-level alignment."""
    sm = SequenceMatcher(None, model_line, gt, autojunk=False)
    g0, g1 = None, None
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if g0 is None and a0 <= i0 < a1:
            g0 = b0 + (i0 - a0 if tag == "equal" else 0)
        if a0 < i1 <= a1:
            g1 = b0 + (i1 - a0) if tag == "equal" else b1
    if g0 is None:
        g0 = 0
    if g1 is None:
        g1 = len(gt)
    return gt[g0:g1]


def verdict(gt_seg: str, ocr: str, base: str) -> str:
    gt_seg, ocr, base = gt_seg.strip(), ocr.strip(), base.strip()
    if gt_seg == ocr:
        return "not-misread (GT=model)"
    if gt_seg.replace(" ", "") == ocr.replace(" ", ""):
        return "not-misread (spacing-only)"
    if gt_seg == base:
        return "misread (GT=edition)"
    if skeleton(gt_seg) == skeleton(ocr):
        return "misread (marks-only)"
    return "misread (other)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--val-dir", type=Path, default=Path("data/processed/annotated_samples/OCR/validation")
    )
    ap.add_argument(
        "--out", type=Path, default=Path("tests/ocr/evaluations/subclass_validation_val300")
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    gts = {
        f.name.removesuffix(".gt.txt"): nfc(f.read_text(encoding="utf-8").strip())
        for f in sorted(a.val_dir.glob("*.gt.txt"))
    }
    print(f"val lines with GT: {len(gts)}")

    for model, root in MODELS.items():
        root = Path(root)
        diffs = json.loads((root / "line_diff.json").read_text())
        rows = []
        n_lines_hit = 0
        for stem, gt in gts.items():
            m = re.match(r"(.+)_line_(\d+)$", stem)
            page, ln = m.group(1), m.group(2)
            dl = [
                d
                for d in diffs.get(page, {}).get("by_line", {}).get(ln, [])
                if d["group"] == "substantive" and d.get("subclass") in SUBCLASSES
            ]
            if not dl:
                continue
            txt_f = root / page / f"{stem}.txt"
            if not txt_f.exists():
                continue
            model_line = nfc(txt_f.read_text(encoding="utf-8").strip())
            n_lines_hit += 1
            for d in dl:
                ocr, base = nfc(d["ocr_text"]), nfc(d["base_text"])
                i0 = model_line.find(ocr)
                if i0 < 0:
                    v, gt_seg = "unlocatable", ""
                else:
                    gt_seg = map_span(model_line, gt, i0, i0 + len(ocr))
                    v = verdict(gt_seg, ocr, base)
                rows.append(
                    {
                        "stem": stem,
                        "subclass": d["subclass"],
                        "type": d["type"],
                        "ocr_text": ocr,
                        "base_text": base,
                        "gt_segment": gt_seg.strip(),
                        "verdict": v,
                    }
                )

        print(f"\n=== {model} — {n_lines_hit} val lines with target diffs, {len(rows)} diffs ===")
        for sub in SUBCLASSES:
            sel = [r for r in rows if r["subclass"] == sub]
            c = Counter(r["verdict"] for r in sel)
            n = len(sel)
            print(f"\n  {sub}  (n={n})")
            for v, k in c.most_common():
                print(f"    {v:<28} {k:>4}  ({100 * k / n:.1f}%)")
            mis = sum(k for v, k in c.items() if v.startswith("misread"))
            notm = sum(k for v, k in c.items() if v.startswith("not-misread"))
            und = n - mis - notm
            print(
                f"    -> misread {mis} ({100 * mis / n:.1f}%) | not-misread {notm} "
                f"({100 * notm / n:.1f}%) | unresolved {und}"
            )

        fname = a.out / f"cases_{model.replace('+', '')}.csv"
        with open(fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  per-case CSV -> {fname}")


if __name__ == "__main__":
    main()
