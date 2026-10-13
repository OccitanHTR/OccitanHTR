"""Does token-level prediction (BPE-150 merges) shape TrOCR's errors and dilute its confidence?

kraken predicts one character per decision; the TrOCR decoder predicts one BPE-150 token
(1..n chars) per decision, and every character of a merge inherits the token's single
probability. Two measurable consequences (spec §5.1.5 / §6.13), on the 300-val:

1. GRANULARITY-INDUCED ERROR STRUCTURE
   - error burstiness: distribution of consecutive-error run lengths, kraken vs TrOCR.
     If wrong merges substitute 2 chars at once, TrOCR should show excess runs >= 2.
   - within-token clustering: error rate for chars inside multi-char tokens vs the base
     rate; among erroneous merges, how often is the WHOLE merge wrong vs one char.

2. CONFIDENCE COST OF SHARED TOKEN PROBABILITIES
   - char-level AUROC / conf separation / ECE computed separately for chars emitted as
     single-char tokens vs chars inside merges. A worse merge-group AUROC quantifies how
     much of the kraken-TrOCR calibration gap is a representation artifact.

    PROJECT_ROOT=. uv run python scripts/ocr/granularity_analysis.py \
        --trocr models/vit_lightreal_med4k/trocr_20260823_073535/best_model \
        --kraken models/ocr/finetuned/finetune_20260806_123435/model_best.mlmodel \
        --val-dir data/processed/annotated_samples/OCR/validation
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confidence_analysis import ece, kraken_preds, label_errors  # noqa: E402


def trocr_preds_tokens(model_dir: Path, crops, device="mps"):
    """Like confidence_analysis.trocr_preds but keeps token boundaries:
    per line -> (pred, confs, tok_len, tok_idx) aligned per character."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

    from src.ocr.image_prep import prepare_image

    resize = "pad"
    if (model_dir / "resize_mode.txt").is_file():
        resize = (model_dir / "resize_mode.txt").read_text().strip() or "pad"
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    out = {}
    for crop in crops:
        img = prepare_image(Image.open(crop).convert("RGB"), proc, resize)
        pv = proc(images=img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            g = model.generate(
                pixel_values=pv,
                num_beams=1,
                max_length=128,
                output_scores=True,
                return_dict_in_generate=True,
            )
        trans = model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True)
        gen_ids = g.sequences[0, -trans.shape[1] :]
        probs = trans[0].exp().tolist()
        chars, confs, tlen, tidx = [], [], [], []
        for j, (tid, pr) in enumerate(zip(gen_ids.tolist(), probs, strict=True)):
            piece = tok.decode([tid], skip_special_tokens=True)
            for ch in piece:
                chars.append(ch)
                confs.append(float(pr))
                tlen.append(len(piece))
                tidx.append(j)
        out[crop.stem] = ("".join(chars), confs, tlen, tidx)
    return out


def error_runs(err: list[bool]) -> list[int]:
    runs, cur = [], 0
    for e in err:
        if e:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trocr", type=Path, required=True)
    ap.add_argument("--kraken", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    crops = sorted(a.val_dir.glob("*.png"))
    gts = {
        c.stem: c.with_name(c.stem + ".gt.txt").read_text(encoding="utf-8").strip() for c in crops
    }
    gts = {k: v for k, v in gts.items() if v}
    crops = [c for c in crops if c.stem in gts]

    print(f"running TrOCR (token-tracked) on {len(crops)} lines ...", flush=True)
    tp = trocr_preds_tokens(a.trocr, crops, a.device)
    print("running kraken ...", flush=True)
    kp = kraken_preds(a.kraken, crops, "cpu")

    # ---- pooled per-char arrays ----
    k_runs, t_runs = [], []
    k_err_n = k_n = 0
    err_all, conf_all, multi_all, tok_all = [], [], [], []  # trocr char-level
    whole_merge = partial_merge = 0
    for stem, gt in gts.items():
        pred, confs = kp.get(stem, ("", []))
        if pred and len(confs) == len(pred):
            e = label_errors(pred, gt)
            k_runs += error_runs(e)
            k_err_n += sum(e)
            k_n += len(e)
        pred, confs, tlen, tidx = tp.get(stem, ("", [], [], []))
        if not pred or len(confs) != len(pred):
            continue
        e = label_errors(pred, gt)
        t_runs += error_runs(e)
        err_all += e
        conf_all += confs
        multi_all += [tl > 1 for tl in tlen]
        tok_all += [(stem, j) for j in tidx]
        # merge-level: for each multi-char token with >=1 error, whole vs partial
        by_tok: dict = {}
        for i, j in enumerate(tidx):
            by_tok.setdefault(j, []).append(i)
        for j, idxs in by_tok.items():
            if len(idxs) > 1 and any(e[i] for i in idxs):
                if all(e[i] for i in idxs):
                    whole_merge += 1
                else:
                    partial_merge += 1

    err = np.array(err_all)
    conf = np.array(conf_all)
    multi = np.array(multi_all)

    # ---- 1a. burstiness ----
    def runstats(runs, n_err, n_all, name):
        runs = np.array(runs)
        frac2 = runs[runs >= 2].sum() / max(1, runs.sum())
        c = Counter(runs.tolist())
        dist = "  ".join(f"len{k}:{c[k]}" for k in sorted(c) if k <= 4) + (
            f"  len≥5:{sum(v for k, v in c.items() if k >= 5)}"
        )
        print(
            f"{name:>8}: {n_err} errors in {len(runs)} runs (rate {n_err / n_all:.4f}) | "
            f"mean run {runs.mean():.2f} | % of error-chars in runs≥2: {100 * frac2:.0f}% | {dist}"
        )
        return frac2

    print("\n=== 1a. error burstiness (consecutive-error run lengths) ===")
    runstats(k_runs, k_err_n, k_n, "kraken")
    runstats(t_runs, int(err.sum()), len(err), "TrOCR")

    # ---- 1b. within-token clustering ----
    base = multi.mean()
    err_in_multi = err[multi].mean()
    err_in_single = err[~multi].mean()
    share_err_multi = multi[err == 1].mean()
    print("\n=== 1b. within-token error clustering (TrOCR) ===")
    print(f"chars carried by multi-char merges: {100 * base:.1f}% (base rate)")
    print(
        f"share of ERRORS inside merges:      {100 * share_err_multi:.1f}%"
        f"  ({'EXCESS' if share_err_multi > base else 'no excess'})"
    )
    print(
        f"char error rate inside merges: {err_in_multi:.4f}  |  single-char tokens: {err_in_single:.4f}"
        f"  (ratio {err_in_multi / max(err_in_single, 1e-9):.2f}x)"
    )
    print(
        f"erroneous merges: whole-merge wrong {whole_merge}  vs  partially wrong {partial_merge}"
        f"  ({100 * whole_merge / max(1, whole_merge + partial_merge):.0f}% all-or-nothing)"
    )

    # ---- 2. confidence split ----
    print("\n=== 2. confidence cost of shared token probabilities (TrOCR char-level) ===")
    print(
        f"{'group':>22} | {'chars':>6} {'errs':>5} | {'conf✓':>6} {'conf✗':>6} | {'AUROC':>6} | {'ECE':>6}"
    )
    for name, m in [
        ("single-char tokens", ~multi),
        ("inside merges", multi),
        ("ALL", np.ones_like(multi, bool)),
    ]:
        e_, c_ = err[m], conf[m]
        auroc = roc_auc_score(e_, 1 - c_) if e_.min() != e_.max() else float("nan")
        e0 = ece(c_, 1 - e_)[0]
        print(
            f"{name:>22} | {len(e_):>6} {int(e_.sum()):>5} | {c_[e_ == 0].mean():>6.3f} "
            f"{c_[e_ == 1].mean():>6.3f} | {auroc:>6.3f} | {e0:>6.4f}"
        )


if __name__ == "__main__":
    main()
