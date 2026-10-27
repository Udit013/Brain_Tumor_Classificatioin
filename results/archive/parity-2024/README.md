# Archived results: parity model (deployed until 2026-10)

Measured on CPU with the serving preprocessing, before the deployed model was
replaced by the fine-tuned model (see the main README). Weights: Hugging Face
tag `parity-2024` of `Udit013/brain-tumor-efficientnetb3`, and locally
`models/archive/parity-2024/` (with SHA256SUMS). To re-measure:

    BTC_WEIGHTS=models/EfficientNetB3_parity_weights.h5 python -m btc.evaluate

`metrics/calibration.json` uses the test-split protocol (this model trained on
all of Kaggle Training, so no clean validation data exists for it);
`calibration_holdout50.json` fits on half the test set and scores the other
half. `external_tta.json` scores the 6-view test-time augmentation average.
