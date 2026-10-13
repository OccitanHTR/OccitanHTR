# OCC-HTR — Transcribing Medieval Occitan

A pipeline that converts photographs of a medieval Occitan manuscript into a line-level
diplomatic transcription — and then systematically compares that transcription with the
modern scholarly edition, classifying every difference (editorial normalization vs.
substantive divergence) with TEI encoding and a human-in-the-loop review interface.

*Chirurgie d'Albucasis* (Montpellier, H 95), 71 double pages, 13,677 text lines.

## Installation

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # locked environment from pyproject.toml + uv.lock
make setup_data            # map the repo's data layout onto the pipeline paths
```

Models: the fine-tuned Kraken recognizer, character LM, and layout detector ship in
`code/models/`. The TrOCR model (~1.1 GB) will be released after review.

## Quick start

```bash
cd code
make frontend                      # web interface at http://localhost:8000
```

Every pipeline stage is a make target — preprocessing (`segment_images`, `crop_segments`,
`filter_images`), training (`kraken_finetune`, `trocr_finetune`), transcription
(`kraken_lm_transcribe`, `trocr_transcribe_conf`), comparison (`align_scholarly_edition`,
`align_transcriptions`, `diff_transcriptions`). See `code/makefile` — it is the executable
documentation.

## Configuration

- **Make variables**: every target's inputs/outputs are variables with defaults, overridable
  inline: `make kraken_lm_transcribe KRAKEN_LM_LAMBDA=0.3`.
- **Frontend**: all paths are `VIEWER_*` environment variables (see `frontend/config.py`) —
  point the viewer at any pipeline run without code changes.
- **Environment**: `pyproject.toml` + `uv.lock` pin everything (no `requirements.txt`);
  `PROJECT_ROOT` controls path resolution outside the default layout.

## Repository structure

```
├── data/              # annotated lines (train/validation/test), sample page, excerpts, lexicon
├── code/              # src/ (library) · scripts/ (CLIs) · makefile · frontend/ · fonts/ · models/
└── results/           # final-model evaluations, full transcriptions, discrepancy catalogues
```

The `data/` folder ships the 900 annotated lines (600 training + 300 held-out test; `validation_data/` holds 100 lines that are a *subset* of the training set, kept aside to tune the language-model weight), one facsimile sample page,
scholarly-edition and corpora excerpts, and the DOM lexicon; full facsimile and edition are
not redistributed (rights) — the excerpts keep all code paths executable.
