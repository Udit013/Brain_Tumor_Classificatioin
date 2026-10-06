"""NeuroClass v2 — EXPERIMENTAL improved training recipe (not yet trained to completion).

STATUS: no v2 model exists yet and no v2 metric is reported anywhere. The first
full run (backbone BatchNorm frozen, learning rate 1e-3 as in the parity recipe)
stayed at chance (27% train / 25% val accuracy after one epoch). 60-step
diagnostics on a 960-image subset isolated the learning rate: with frozen
BatchNorm nothing re-normalises activations, and 1e-3 scrambles the pretrained
features. At 1e-4 the same setup reached 56% train / 65% val accuracy in 60
steps (parity recipe on CPU: 84% / 64%). The default below is therefore 1e-4;
a full run is the next step.

train.py reproduces the published recipe exactly (parity). This module is the
separate, improved recipe. Every change targets a weakness measured on the
parity model:

  problem measured on the parity model           change here
  ---------------------------------------------  -----------------------------------------
  checkpoint chosen by accuracy on the TEST set  selection on a held-out validation split
                                                 (btc.splits: near-duplicate clusters never
                                                 straddle train/val)
  trained on Metal, which computes a different   trained on CPU — the computation that is
  model than the deployed CPU (93.94 vs 91.25%)  evaluated and served
  train/serve resize skew (91.25 vs 90.25%)      btc.preprocess for training, eval and serving
  BatchNorm running stats never converged        backbone BatchNorm frozen (standard practice
  (inference collapsed to 53% before a fix)      for fine-tuning EfficientNet): it keeps its
                                                 ImageNet statistics, so train and inference
                                                 compute the same function
  steps_per_epoch quirk: each epoch saw ~half    full passes over the training split
  the training data
  near-chance accuracy under Gaussian noise      augmentation: flips, small rotations/zooms/
                                                 shifts, brightness/contrast, Gaussian noise

The architecture is unchanged (same layers, same weight layout), so the ONNX
export, the FastAPI service and the Hugging Face Space only need new weights.

Outputs:
  models/NeuroClass_v2_weights.h5
  results/metrics/train_v2_history.json

Usage:
    python -m btc.train_v2 [--epochs 15]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import config

V2_WEIGHTS_PATH = config.V2_WEIGHTS_PATH
NOISE_PROB = 0.5        # fraction of training images that receive Gaussian noise
NOISE_MAX_STD = 25.0    # per-image noise std ~ U(0, 25) on the 0-255 scale
V2_LEARNING_RATE = 1e-4  # 1e-3 (parity) does not learn with frozen BN — see STATUS


def _augment_fn():
    import tensorflow as tf

    geom = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(10 / 360, fill_mode="reflect"),
        tf.keras.layers.RandomZoom(0.1, fill_mode="reflect"),
        tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
    ])

    def augment(x, y):
        x = geom(x, training=True)
        x = tf.image.random_brightness(x, max_delta=25.0)
        x = tf.image.random_contrast(x, 0.8, 1.2)
        std = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.0, NOISE_MAX_STD)
        apply = tf.cast(tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < NOISE_PROB, tf.float32)
        x = x + apply * std * tf.random.normal(tf.shape(x))
        return tf.clip_by_value(x, 0.0, 255.0), y

    return augment


def build_v2_model():
    """Parity architecture with the backbone's BatchNorm layers frozen."""
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.optimizers import Adamax

    from .model import build_model

    model = build_model(weights="imagenet")
    base = model.get_layer("efficientnetb3")
    n = 0
    for layer in base.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False   # TF2: a non-trainable BN runs in inference mode
            n += 1
    model.compile(optimizer=Adamax(learning_rate=V2_LEARNING_RATE),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model, n


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NeuroClass v2")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    config.configure_inference_device()   # train on CPU (see module docstring)
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    from .preprocess import load_images
    from .splits import load_split

    tf.keras.utils.set_random_seed(config.SEED)
    config.ensure_dirs()
    split = load_split()
    t0 = time.time()
    x_tr = load_images(split["train"][0]).astype("uint8")
    x_va = load_images(split["val"][0])
    y_tr = tf.one_hot(split["train"][1], config.NUM_CLASSES)
    y_va = tf.one_hot(split["val"][1], config.NUM_CLASSES)
    print(f"loaded {len(x_tr)} train / {len(x_va)} val images in {time.time()-t0:.0f}s")

    train_ds = (tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
                .shuffle(len(x_tr), seed=config.SEED, reshuffle_each_iteration=True)
                .map(lambda x, y: (tf.cast(x, tf.float32), y))
                .batch(config.BATCH_SIZE)
                .map(_augment_fn(), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = tf.data.Dataset.from_tensor_slices((x_va, y_va)).batch(32)

    model, n_frozen = build_v2_model()
    print(f"froze {n_frozen} backbone BatchNorm layers; "
          f"trainable params {sum(int(np.prod(w.shape)) for w in model.trainable_weights):,}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[
            ModelCheckpoint(str(V2_WEIGHTS_PATH), monitor="val_accuracy", mode="max",
                            save_best_only=True, save_weights_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, verbose=1),
            EarlyStopping(monitor="val_accuracy", mode="max", patience=4,
                          restore_best_weights=True, verbose=1),
        ],
        verbose=2,
    )
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist["best_val_accuracy"] = max(hist["val_accuracy"])
    hist["best_epoch"] = int(np.argmax(hist["val_accuracy"])) + 1
    hist["train_seconds"] = round(time.time() - t0)
    (config.METRICS_DIR / "train_v2_history.json").write_text(json.dumps(hist, indent=2))
    print(f"best val_accuracy {hist['best_val_accuracy']:.4f} at epoch {hist['best_epoch']} "
          f"-> {V2_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
