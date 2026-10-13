# Deployed model artifacts

- `ocr/kraken_finetuned_leader.mlmodel` — the fine-tuned Kraken CRNN–CTC recognizer (checkpoint `finetune_20260806_123435`, the
  leader model; 97.44%/83.71% char/word accuracy with LM rescoring on the test set).
- `lm/lm_600.pkl`, `lm/lm_600_medical.pkl` — character 6-gram language models for rescoring
  (stupid-backoff pickles; `lm_600` is the deployed one, λ = 0.2).
- `layout/y8_YALTAi_15_best_+9annotated_fix50.pt` — YOLOv8n MainZone detector (deployed).

Not included here:
- **TrOCR (ViT+RoBERTa) final model** (~1.1 GB): will be released after review.
- **kraken blla segmentation model**: stock model shipped with kraken (downloaded automatically).
- **CatMuS Medieval 1.5.0 baseline**: obtain from the official Zenodo/HuggingFace release
  (Pinche et al., 2024).
