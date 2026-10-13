# OCR evaluation — krakenLM_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `krakenLM` from `data/processed/transcription/krakenLM_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| krakenLM | 299 | 0.0256 | 0.9744 | 0.1629 | 0.8371 | 0.0238 | 0.1250 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
