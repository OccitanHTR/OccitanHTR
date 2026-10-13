"""Re-evaluate the 6-arch × 4-scenario encoder/decoder grid under the clipped convention.

The §6.5.15 grid (spec) was populated before per-line clipping (edit distance capped at the
reference length) became the project-wide convention; the from-scratch architectures
over-produce heavily, so clipping moves some of their cells by several pp. This recomputes
char_acc AND word_acc for all 24 cells from the stored val-300 transcriptions with the
current `evaluate_ocr` (clipped, NFC). GPT-2 cells use the fixed `*_v2` runs (§6.5.7).

    PROJECT_ROOT=. uv run python scripts/ocr/grid24_clipped_eval.py
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.ocr.evaluate_ocr import eval_one_model, normalise

CELLS: dict[str, dict[str, str]] = {
    "ViT+RoBERTa": {
        "600only": "fill_vitroberta_600only_val300",
        "600+3000": "fill_vitroberta_600_3000_val300",
        "A_cometa": "vitroberta_matched_cometa_val300_20260722",
        "B_medical": "vitroberta_medical_val300_20260722",
    },
    "Swin+xlm-RoBERTa": {
        "600only": "fill_swinroberta_600only_val300",
        "600+3000": "fill_swinroberta_600_3000_val300",
        "A_cometa": "swinroberta_A_val300_20260724",
        "B_medical": "swinroberta_B_val300_20260724",
    },
    "Swin+GPT2": {
        "600only": "fill_swingpt2_600only_val300",
        "600+3000": "fill_swingpt2_600_3000_val300",
        "A_cometa": "swingpt2_A_v2_val300_20260724",
        "B_medical": "swingpt2_B_v2_val300_20260724",
    },
    "ViT+GPT2": {
        "600only": "fill_vitgpt2_600only_val300",
        "600+3000": "fill_vitgpt2_600_3000_val300",
        "A_cometa": "vitgpt2_A_v2_val300_20260724",
        "B_medical": "vitgpt2_B_v2_val300_20260724",
    },
    "ViT+BERT": {
        "600only": "fill_vitbert_600only_val300",
        "600+3000": "fill_vitbert_600_3000_val300",
        "A_cometa": "vitbert_A_val300_20260724",
        "B_medical": "vitbert_B_val300_20260724",
    },
    "Swin+BERT": {
        "600only": "fill_swinbert_600only_val300",
        "600+3000": "fill_swinbert_600_3000_val300",
        "A_cometa": "swinbert_ss_matched_cometa_val300_20260722",
        "B_medical": "swinbert_ss_medical_val300_20260722",
    },
}


def main() -> None:
    gt_dir = Path("data/processed/annotated_samples/OCR/validation")
    gt = {
        f.name.removesuffix(".gt.txt"): normalise(f.read_text(encoding="utf-8"))
        for f in sorted(gt_dir.glob("*.gt.txt"))
    }
    log = logging.getLogger("grid24")
    log.addHandler(logging.NullHandler())
    log.propagate = False
    base = Path("data/processed/transcription")
    out = Path("tests/ocr/evaluations/grid24_clipped_20260914")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"{'arch':<18} {'scenario':<9} | {'char_acc':>8} {'word_acc':>8} | n / miss")
    for arch, sc in CELLS.items():
        for name, d in sc.items():
            _, agg = eval_one_model(arch, base / d, gt, log)
            rows.append(
                {
                    "arch": arch,
                    "scenario": name,
                    "transcription_dir": d,
                    "char_acc": round(agg["char_acc_corpus"], 4),
                    "word_acc": round(agg["word_acc_corpus"], 4),
                    "n_lines": agg["n_lines_scored"],
                    "n_missing": agg["n_skipped_missing_pred"],
                }
            )
            r = rows[-1]
            print(
                f"{arch:<18} {name:<9} | {r['char_acc']:>8.4f} {r['word_acc']:>8.4f} "
                f"| {r['n_lines']} / {r['n_missing']}"
            )

    with open(out / "grid24_clipped.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV -> {out}/grid24_clipped.csv")


if __name__ == "__main__":
    main()
