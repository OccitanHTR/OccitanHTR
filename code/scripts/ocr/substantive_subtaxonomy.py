"""Sub-segment the SUBSTANTIVE discrepancies: manuscript reading vs probable misread.

The substantive group (spec §6.7.2) mixes scribal conventions, genuine variants
and model misreads. This prototype splits it using three independent signals:

  a. cross-model corroboration — the OTHER leader produced the SAME reading
     against the edition (two architectures agreeing on the same deviation ->
     the scribe almost certainly wrote it);
  b. lexicon attestation — the OCR-side words are attested Old Occitan forms
     (DOM lemma variants, 36.7k forms); a non-word is almost certainly a misread;
  c. model confidence — min per-char/token confidence over the diff span
     (validated error flag, spec §6.13); low confidence leans misread.

Sub-taxonomy per substantive diff:
  scribal-convention  spacing/abbreviation pair occurring >= --systematic-n times
  attested-variant    corroborated AND (attested or spacing/abbreviation)
  probable-misread    NOT corroborated AND (non-word OR min-conf < --low-conf)
  uncertain           everything else

Usage:
    PROJECT_ROOT=. uv run python scripts/ocr/substantive_subtaxonomy.py \
        --out tests/ocr/evaluations/substantive_subtaxonomy_20260907
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from src.ocr.line_diff import _fold

MODELS = {
    "kraken": ("krakenLM_full_corpus", "krakenleader_conf_fullms"),
    "trocr": ("vitlightreal_full_corpus", "vitlightreal_conf_fullms"),
}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def load_diffs(diff_json: Path) -> list[dict]:
    d = json.loads(diff_json.read_text(encoding="utf-8"))
    out = []
    for page, pg in d.items():
        for ln, diffs in pg["by_line"].items():
            for pos, x in enumerate(diffs):
                if x["group"] == "substantive":
                    # pos = index in the by_line list, so --write-back can
                    # address exactly this diff in the source JSON
                    out.append({**x, "page": page, "line": ln, "pos": pos})
    return out


def write_back(diff_json: Path, labels: list[tuple[str, str, int, str]]) -> None:
    """Annotate the source line_diff.json in place: ``subclass`` on each
    substantive diff (scribal-convention / attested-variant / probable-misread
    / uncertain). Non-substantive diffs are untouched."""
    d = json.loads(diff_json.read_text(encoding="utf-8"))
    for page, ln, pos, cls in labels:
        d[page]["by_line"][ln][pos]["subclass"] = cls
    diff_json.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def conf_lookup(conf_dir: Path):
    """stem -> (text, per-char conf list) built lazily per page."""
    cache: dict[str, dict] = {}

    def get(page: str, stem: str):
        if page not in cache:
            p = conf_dir / f"{page}.json"
            cache[page] = json.loads(p.read_text(encoding="utf-8"))["lines"] if p.is_file() else {}
        rec = cache[page].get(stem)
        if rec is None:
            return None, None
        pieces = rec.get("chars") or rec.get("tokens") or []
        chars, confs = [], []
        for piece, c in pieces:
            for ch in piece:
                chars.append(ch)
                confs.append(float(c))
        return "".join(chars), confs

    return get


def span_min_conf(get_conf, page: str, line: str, ocr_text: str) -> float | None:
    text, confs = get_conf(page, f"{page}_line_{line}")
    if not text or not ocr_text:
        return None
    pos = text.find(ocr_text)
    if pos < 0:  # e.g. kraken: LM-rescored diff text vs raw-CTC conf text
        return None
    window = [c for ch, c in zip(text[pos : pos + len(ocr_text)], confs[pos : pos + len(ocr_text)]) if not ch.isspace()]
    return min(window) if window else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dom", type=Path, default=Path("data/raw/DOM_lemma_variants.json"))
    ap.add_argument("--tx-base", type=Path, default=Path("data/processed/transcription"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--low-conf", type=float, default=0.90)
    ap.add_argument("--systematic-n", type=int, default=10)
    ap.add_argument(
        "--write-back", action="store_true",
        help="annotate each model's line_diff.json in place with a `subclass` field",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lex = {k.lower() for k in json.loads(args.dom.read_text(encoding="utf-8"))}
    lex_folded = {_fold(k) for k in lex}

    diffs = {m: load_diffs(args.tx_base / d / "line_diff.json") for m, (d, _) in MODELS.items()}
    conf = {m: conf_lookup(args.tx_base / c) for m, (_, c) in MODELS.items()}
    # corroboration index: (page, folded base, folded ocr) of the OTHER model
    idx = {m: {(x["page"], _fold(x["base_text"]), _fold(x["ocr_text"])) for x in xs} for m, xs in diffs.items()}
    other = {"kraken": "trocr", "trocr": "kraken"}

    def attested(ocr_text: str) -> bool | None:
        words = _WORD.findall(ocr_text.lower())
        if not words:
            return None
        return all(w in lex or _fold(w) in lex_folded for w in words)

    for m, xs in diffs.items():
        pair_n = Counter((x["base_text"], x["ocr_text"], x["type"]) for x in xs)
        classes = Counter()
        conf_by_class: dict[str, list[float]] = {}
        rows = []
        labels: list[tuple[str, str, int, str]] = []
        for x in xs:
            corr = (x["page"], _fold(x["base_text"]), _fold(x["ocr_text"])) in idx[other[m]]
            att = attested(x["ocr_text"])
            mc = span_min_conf(conf[m], x["page"], x["line"], x["ocr_text"])
            systematic = pair_n[(x["base_text"], x["ocr_text"], x["type"])] >= args.systematic_n
            if x["type"] in ("spacing", "abbreviation") and systematic:
                cls = "scribal-convention"
            elif corr and (att or x["type"] in ("spacing", "abbreviation")):
                cls = "attested-variant"
            elif not corr and (att is False or (mc is not None and mc < args.low_conf)):
                cls = "probable-misread"
            else:
                cls = "uncertain"
            classes[cls] += 1
            labels.append((x["page"], x["line"], x["pos"], cls))
            if mc is not None:
                conf_by_class.setdefault(cls, []).append(mc)
            rows.append([x["page"], x["line"], x["type"], x["base_text"], x["ocr_text"],
                         int(corr), "" if att is None else int(att),
                         "" if mc is None else round(mc, 4), cls])

        total = len(xs)
        print(f"\n== {m}: {total} substantive diffs")
        for cls, n in classes.most_common():
            cs = conf_by_class.get(cls, [])
            mean_c = f"{sum(cs)/len(cs):.3f}" if cs else "  -  "
            print(f"   {cls:<20} {n:>6}  {n/total*100:5.1f}%   mean min-conf {mean_c}")
        with open(args.out / f"subtaxonomy_{m}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["page", "line", "type", "scholarly", "model", "corroborated",
                        "attested", "min_conf", "class"])
            w.writerows(rows)
        print(f"   -> {args.out}/subtaxonomy_{m}.csv")
        if args.write_back:
            path = args.tx_base / MODELS[m][0] / "line_diff.json"
            write_back(path, labels)
            print(f"   subclass written back into {path}")


if __name__ == "__main__":
    main()
