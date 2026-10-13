"""Map the submission-repo layout onto the paths the pipeline expects.

The repository follows a template (data/train_data, data/test_data, ...),
while the makefile and scripts expect the original project layout under code/data and
code/models. Run this once after cloning:

    cd code && python scripts/setup_data_layout.py

It creates symlinks (falling back to copies where symlinks are unavailable), so the
make targets run against the shipped sample data without any path edits.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import os

CODE = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("TEMPLATE_DATA_ROOT", CODE.parent))  # repo root (or mount)

MAPPINGS = [
    # (link inside code/, target in the template layout)
    ("data/processed/annotated_samples/OCR/full_annotated", "data/train_data"),
    ("data/processed/annotated_samples/OCR/validation", "data/test_data"),
    ("data/processed/annotated_samples/OCR/dev100_20260815", "data/validation_data"),
    ("data/raw/original_manuscript/reproduction14453_100", "data/sample_page"),
    ("data/raw/AlbucE.txt", "data/scholarly_edition/AlbucE_excerpt.txt"),
    ("data/raw/COMETA_medieval_corpus", "data/corpora/COMETA_excerpt"),
    ("data/raw/medical_texts", "data/corpora/Medical_excerpt"),
    ("data/raw/DOM_lemma_variants.json", "data/lexicon/DOM_lemma_variants.json"),
    # deployed kraken model under the path the makefile defaults to
    ("models/ocr/finetuned/finetune_20260806_123435/model_best.mlmodel",
     "code/models/ocr/kraken_finetuned_leader.mlmodel"),
    # character-LM pickles where the rescoring scripts look for them
    ("data/processed/lm_corpora/lm_600.pkl", "code/models/lm/lm_600.pkl"),
    ("data/processed/lm_corpora/lm_600_medical.pkl", "code/models/lm/lm_600_medical.pkl"),
]


def link(dst: Path, src: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return "exists"
    try:
        import os
        dst.symlink_to(os.path.relpath(src, dst.parent))
        return "linked"
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return "copied"


def main() -> None:
    for rel_dst, rel_src in MAPPINGS:
        src = REPO / rel_src
        dst = CODE / rel_dst
        if not src.exists():
            print(f"  MISSING source, skipped: {rel_src}")
            continue
        print(f"  {link(dst, src):7s} {rel_dst}  ->  {rel_src}")
    print("Done. Note: the shipped data are the annotated lines plus excerpts/sample page;")
    print("full-corpus targets need the complete facsimile and edition (see README).")


if __name__ == "__main__":
    main()
