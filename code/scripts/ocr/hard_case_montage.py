"""Montage images of the hard-case funnel groups (spec §6.13) for visual comparison.

For each group of the funnel — hard for BOTH (11), kraken-only (19), TrOCR-only (19) — stack the
actual validation line crops with, under each: the ground truth and both models' readings with
their per-line CER. Makes the complementary difficulty profiles (physical bleed vs content
ambiguity) inspectable at a glance.

    PROJECT_ROOT=. uv run python scripts/ocr/hard_case_montage.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

W = 1180  # content width
PAD = 20
CROP_W = W - 2 * PAD


def fonts():
    reg = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 15)
    bold = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), 16)
    mono = ImageFont.truetype(font_manager.findfont("DejaVu Sans Mono"), 14)
    title = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), 22)
    return reg, bold, mono, title


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--features",
        type=Path,
        default=Path("tests/ocr/evaluations/longtail_confidence/hard_case_features.csv"),
    )
    ap.add_argument(
        "--longtail-dir", type=Path, default=Path("tests/ocr/evaluations/longtail_confidence")
    )
    ap.add_argument(
        "--val-dir", type=Path, default=Path("data/processed/annotated_samples/OCR/validation")
    )
    ap.add_argument("--out", type=Path, default=Path("tests/ocr/evaluations/longtail_confidence"))
    a = ap.parse_args()
    reg, bold, mono, title_f = fonts()

    rows = list(csv.DictReader(open(a.features, encoding="utf-8")))
    preds = {}
    for m in ("kraken", "trocr"):
        preds[m] = {
            r["stem"]: r["pred"]
            for r in csv.DictReader(open(a.longtail_dir / f"per_line_{m}.csv", encoding="utf-8"))
        }
    gts = {
        r["stem"]: r["gt"]
        for r in csv.DictReader(open(a.longtail_dir / "per_line_kraken.csv", encoding="utf-8"))
    }

    groups = {
        "BOTH": (
            "hard_cases_both.jpg",
            "Hard for BOTH models",
            lambda r: -(float(r["kraken_cer"]) + float(r["trocr_cer"])),
        ),
        "kraken-only": (
            "hard_cases_kraken_only.jpg",
            "Hard for kraken ONLY",
            lambda r: -float(r["kraken_cer"]),
        ),
        "trocr-only": (
            "hard_cases_trocr_only.jpg",
            "Hard for TrOCR ONLY",
            lambda r: -float(r["trocr_cer"]),
        ),
    }
    for gname, (fname, gtitle, keyfn) in groups.items():
        sel = sorted((r for r in rows if r["group"] == gname), key=keyfn)
        # ---- measure heights ----
        entries = []
        for r in sel:
            stem = r["stem"]
            img = Image.open(a.val_dir / f"{stem}.png").convert("RGB")
            scale = CROP_W / img.width
            ch = max(24, int(img.height * scale))
            entries.append((r, img, ch))
        line_h = 20

        def entry_h(ch: int) -> int:  # caption + crop + 3 text rows + gap
            return 24 + ch + 3 * line_h + 14

        H = 64 + sum(entry_h(ch) for _, _, ch in entries) + PAD
        canvas = Image.new("RGB", (W, H), "#ffffff")
        d = ImageDraw.Draw(canvas)
        d.text((PAD, 18), f"{gtitle}  (n={len(entries)})", font=title_f, fill="#222222")
        y = 64
        for r, img, ch in entries:
            stem = r["stem"]
            kc, tc = float(r["kraken_cer"]), float(r["trocr_cer"])
            cap = f"{stem}   kraken CER {kc:.3f}   ·   TrOCR CER {tc:.3f}   ·   bleed {float(r['bleed']):.2f}"
            d.text((PAD, y), cap, font=bold, fill="#8a2620")
            y += 24
            canvas.paste(img.resize((CROP_W, ch)), (PAD, y))
            d.rectangle([PAD - 1, y - 1, PAD + CROP_W, y + ch], outline="#cccccc")
            y += ch + 2
            for lab, txt, col in (
                ("GT    ", gts.get(stem, ""), "#222222"),
                ("kraken", preds["kraken"].get(stem, ""), "#555555"),
                ("TrOCR ", preds["trocr"].get(stem, ""), "#555555"),
            ):
                d.text((PAD, y), f"{lab}  {txt}", font=mono, fill=col)
                y += line_h
            y += 14
        out = a.out / fname
        canvas.save(out, quality=88, dpi=(150, 150))
        print(f"saved {out}  ({len(entries)} lines, {W}x{H})")


if __name__ == "__main__":
    main()
