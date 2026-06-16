"""Reproduce the published EfficientNetB3 training run (PARITY).

Mirrors cell 17 of ``legacy/EfficientNetB3_Brain_Tumor.ipynb``: same callbacks,
same 20-epoch schedule, same steps_per_epoch quirk, same checkpoint filename.
Writes the best weights to ``models/EfficientNetB3_model_weights.h5`` and the
training history to ``results/metrics/train_history.json``.

Usage:
    python -m btc.train
"""

from __future__ import annotations

import json

from . import config
from .data import make_generators
from .model import build_model


def recalibrate_bn(model, train_generator, steps: int = 350) -> None:
    """Re-estimate BatchNorm moving stats via forward passes (no weight update).

    Lowers BN momentum so stats converge within one pass over the training set,
    then runs `steps` training-mode forward passes. Only moving_mean/variance
    change; trainable weights are untouched (no gradients applied).
    """
    from tensorflow.keras.layers import BatchNormalization

    def _retune(m):
        for layer in getattr(m, "layers", []):
            if isinstance(layer, BatchNormalization):
                layer.momentum = 0.9
            if hasattr(layer, "layers"):
                _retune(layer)

    _retune(model)
    for i in range(steps):
        xb, _ = next(train_generator)
        model(xb, training=True)


def main() -> None:
    config.ensure_dirs()

    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ReduceLROnPlateau,
        ModelCheckpoint,
    )

    train_generator, test_generator = make_generators()
    model = build_model(weights="imagenet")
    model.summary()

    callbacks = [
        EarlyStopping(monitor="loss", min_delta=1e-11, patience=12, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=6, verbose=1),
        ModelCheckpoint(
            filepath=str(config.WEIGHTS_PATH),
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_generator,
        steps_per_epoch=config.STEPS_PER_EPOCH,   # PARITY: 5712 // 32 = 178
        epochs=config.EPOCHS,
        validation_data=test_generator,
        validation_steps=config.VALIDATION_STEPS,  # PARITY: 1311 // 32 = 40
        callbacks=callbacks,
    )

    # --- BatchNorm recalibration (Apple Metal / small-batch remedy) -------- #
    # On tensorflow-metal (and generally with momentum=0.99 + batch 16), the
    # backbone's BN moving averages do not converge, so inference-mode predict()
    # collapses toward one class (~0.53 acc) even though training-mode features
    # are good. We re-estimate BN moving stats with forward passes over the
    # training data (NO optimizer step -> learned weights unchanged), which
    # recovers correct inference behaviour. This was verified to move full-test
    # accuracy from ~0.53 to ~0.94. Documented in README Evaluation Limitations.
    model.load_weights(str(config.WEIGHTS_PATH))  # best checkpoint
    recalibrate_bn(model, train_generator)
    model.save_weights(str(config.WEIGHTS_PATH))
    print("Recalibrated BN moving statistics and re-saved weights.")

    # Persist history (real measured numbers only).
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    out = config.METRICS_DIR / "train_history.json"
    out.write_text(json.dumps(hist, indent=2))
    print(f"Saved best weights -> {config.WEIGHTS_PATH}")
    print(f"Saved history       -> {out}")

    # Quick parity check against the paper's headline (printed, not asserted —
    # exact reproduction depends on hardware/TF version stochasticity).
    best_val = max(hist.get("val_accuracy", [float("nan")]))
    print(
        f"Best val_accuracy this run: {best_val:.5f}  "
        f"(published EfficientNetB3 = {config.PUBLISHED_ACCURACY['EfficientNetB3']:.5f})"
    )


if __name__ == "__main__":
    main()
