"""NeuroClass v2 — EXPERIMENTAL improved training recipe (two-stage fine-tune).

train.py reproduces the published recipe exactly (parity). This module is the
separate, improved recipe. Every change targets a weakness measured on the
parity model:

  problem measured on the parity model           change here
  ---------------------------------------------  -----------------------------------------
  checkpoint chosen by accuracy on the TEST set  selection on a held-out validation split
                                                 (btc.splits: near-duplicate clusters never
                                                 straddle train/val)
  trained via Metal graph mode, which computes   trained on CPU by default (--device cpu);
  a different function than the deployed CPU    the Metal workaround applies with --device
  model (93.81 vs 90.25%)                        gpu, but Metal gradients are unverified
  train/serve resize skew (91.25 vs 90.25%)      btc.preprocess for training, eval and serving
  BatchNorm running stats never converged        backbone BatchNorm frozen (standard practice
  (inference collapsed to 53% before a fix)      for fine-tuning EfficientNet): it keeps its
                                                 ImageNet statistics, so train and inference
                                                 compute the same function
  steps_per_epoch quirk: each epoch saw ~half    full passes over the training split
  the training data
  near-chance accuracy under Gaussian noise      augmentation: flips, small rotations/zooms/
                                                 shifts, brightness/contrast, Gaussian noise

Recipe (the standard way to fine-tune EfficientNet, and much cheaper on CPU):
  stage 1  head only, on cached backbone features (ImageNet backbone frozen;
           features of each training image and its horizontal flip), Adamax 1e-3,
           early-stopped on validation accuracy. Seconds once features are cached.
  stage 2  unfreeze the backbone from --unfreeze-from (default block6a; BatchNorm
           stays frozen) and fine-tune end to end with augmentation at a low
           learning rate. Backpropagation stops at block6a, whose maps are 16x16,
           so a step costs a fraction of full fine-tuning.

An earlier variant unfroze the whole backbone at once: at lr 1e-3 it stayed at
chance (the overfit test in scripts/v2_sanity_checks.py reproduces this: 31% on
32 images vs 100% at 1e-4), and at 1e-4 on CPU it costs ~6 min per epoch.

The architecture is unchanged (same layers, same weight layout), so the ONNX
export, the FastAPI service and the Hugging Face Space only need new weights.

Outputs (never committed, never written to models/):
  runs/<run-name>/metadata.json         config, seed, split hash, git rev, hardware, result
  runs/<run-name>/features.npz          cached stage-1 backbone features
  runs/<run-name>/history_head.csv      stage-1 per-epoch history
  runs/<run-name>/head.weights.h5       model after stage 1
  runs/<run-name>/history_finetune.csv  stage-2 per-epoch history
  runs/<run-name>/best.weights.h5       best stage-2 val_accuracy checkpoint
  runs/<run-name>/{best,head}_serving.h5  the same weights for the serving architecture
                                        (see export_for_serving) — evaluate THESE
  runs/<run-name>/backup/               stage-2 resume state (weights + optimizer + epoch)

Re-running with the same --run-name resumes: cached features and the stage-1
head are reused, and stage 2 continues from its last completed epoch.

A candidate is promoted to models/NeuroClass_v2_weights.h5 by hand, and only
after scripts/evaluate_candidate.py shows it is better on validation and, once,
on the test set. Run scripts/v2_sanity_checks.py before any long job.

Usage:
    python -m btc.train_v2 [--epochs 6] [--unfreeze-from block6a] [--run-name NAME]
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
V2_LEARNING_RATE = 1e-4  # stage 2; 1e-3 destroys pretrained features with BN frozen
HEAD_LEARNING_RATE = 1e-3


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


def build_v2_model(lr: float = V2_LEARNING_RATE):
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
    model.compile(optimizer=Adamax(learning_rate=lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model, n


def _git_rev() -> str:
    import subprocess
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=config.REPO_ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=config.REPO_ROOT) != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 — metadata only
        return "unknown"


def _load_uint8(paths, chunk: int = 256) -> np.ndarray:
    """Serving preprocessing, stored as uint8 (lossless: PIL output is 8-bit)."""
    from .preprocess import load_images

    out = np.empty((len(paths), *config.IMG_SHAPE), dtype=np.uint8)
    for i in range(0, len(paths), chunk):
        out[i:i + chunk] = load_images(paths[i:i + chunk]).astype(np.uint8)
    return out


def _features(base, x: np.ndarray, chunk: int = 256) -> np.ndarray:
    return np.concatenate([base.predict(x[i:i + chunk].astype("float32"), batch_size=32,
                                        verbose=0) for i in range(0, len(x), chunk)])


IMAGENET_STDDEV_RGB = np.array([0.229, 0.224, 0.225], dtype="float32")


def export_for_serving(model, path, probe: np.ndarray) -> float:
    """Write v2 weights loadable by btc.model.load_trained_model (weights=None).

    Keras's EfficientNet inserts an extra input layer,
    Rescaling(1 / sqrt(IMAGENET_STDDEV_RGB)), only when built with
    weights="imagenet" (tensorflow#49930). v2 trains from ImageNet weights, so
    it has that layer; evaluation, ONNX export, FastAPI and the Space build with
    weights=None and do not. Loading v2 weights there silently computes a
    different function (measured: 95.0% -> 86.7% validation accuracy). The
    layer is a per-channel scale right after the Normalization layer, so it
    folds exactly into that layer's variance: (x - m) / sqrt(v) / sqrt(s) =
    (x - m) / sqrt(v * s). The parity model never had the layer at inference
    (its BatchNorm statistics were re-estimated through load_trained_model), so
    it needs no fold — and must not be given one (22.75% if it is).

    Refuses to write unless the exported model reproduces `model` on `probe`.
    """
    from .model import build_model

    serving = build_model(weights=None)
    serving.set_weights(model.get_weights())
    base = serving.get_layer("efficientnetb3")
    norm = next(layer for layer in base.layers if layer.name.startswith("normalization"))
    mean, var, count = norm.get_weights()
    norm.set_weights([mean, var * IMAGENET_STDDEV_RGB, count])
    diff = float(np.abs(model.predict(probe, verbose=0)
                        - serving.predict(probe, verbose=0)).max())
    if diff > 1e-4:
        raise RuntimeError(f"serving export differs from the trained model ({diff:.2e})")
    serving.save_weights(str(path))
    return diff


def unfreeze_from(model, block: str) -> int:
    """Make backbone layers from `block` onward trainable, BatchNorm excepted."""
    from tensorflow.keras.layers import BatchNormalization

    base = model.get_layer("efficientnetb3")
    base.trainable = True
    on, n = False, 0
    for layer in base.layers:
        on = on or layer.name.startswith(block)
        layer.trainable = on and not isinstance(layer, BatchNormalization)
        n += layer.trainable
    if not on:
        raise ValueError(f"no backbone layer starts with {block!r}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NeuroClass v2 (two-stage)")
    parser.add_argument("--head-epochs", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=6, help="stage-2 epochs")
    parser.add_argument("--lr", type=float, default=V2_LEARNING_RATE, help="stage-2 lr")
    parser.add_argument("--unfreeze-from", default="block6a")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-head-bn", action="store_true",
                        help="let the head BatchNorm keep updating in stage 2 (default: "
                             "frozen with its stage-1 statistics from clean images)")
    parser.add_argument("--max-train", type=int, default=None,
                        help="use a random N-image subset of training (smoke tests)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                        help="cpu (default): the computation that is evaluated and "
                             "served. gpu: use a visible accelerator; on Apple Metal the "
                             "graph-mode defect workaround is applied, but gradients are "
                             "not verified equal to CPU (see config."
                             "disable_metal_graph_optimizer). Evaluation is always CPU.")
    parser.add_argument("--run-name", default=None,
                        help="experiment directory under runs/ (re-use it to resume)")
    args = parser.parse_args()

    import hashlib
    import platform

    import tensorflow as tf
    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    else:
        config.disable_metal_graph_optimizer()
    from tensorflow.keras.callbacks import (BackupAndRestore, CSVLogger, EarlyStopping,
                                            ModelCheckpoint, ReduceLROnPlateau)
    from tensorflow.keras.layers import BatchNormalization, Input
    from tensorflow.keras.optimizers import Adamax

    from .splits import SPLIT_PATH, load_split

    # Experiments live outside tracked files and never touch models/.
    run_dir = config.REPO_ROOT / "runs" / (args.run_name or time.strftime("v2-%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(config.SEED)
    gpus = tf.config.get_visible_devices("GPU")
    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "recipe": "train_v2 two-stage", "git_rev": _git_rev(), "seed": config.SEED,
        "head_lr": HEAD_LEARNING_RATE, "head_epochs_max": args.head_epochs,
        "finetune_lr": args.lr, "finetune_epochs_max": args.epochs,
        "unfreeze_from": args.unfreeze_from, "batch_size": args.batch_size,
        "max_train": args.max_train, "train_head_bn": args.train_head_bn,
        "noise_prob": NOISE_PROB, "noise_max_std": NOISE_MAX_STD,
        "split_sha256": hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest()[:16],
        "device": "gpu" if gpus else "cpu", "machine": platform.machine(),
        "tensorflow": tf.__version__,
    })
    meta.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%S"))

    def save_meta():
        meta_path.write_text(json.dumps(meta, indent=2))

    save_meta()
    split = load_split()
    tr_paths, tr_labels = split["train"]
    if args.max_train:   # stratified random subset (the split is ordered by class)
        keep = np.sort(np.random.default_rng(config.SEED).permutation(len(tr_paths))
                       [:args.max_train])
        tr_paths = [tr_paths[i] for i in keep]
        tr_labels = [tr_labels[i] for i in keep]
    t0 = time.time()
    x_tr, x_va = _load_uint8(tr_paths), _load_uint8(split["val"][0])
    y_tr = np.eye(config.NUM_CLASSES, dtype="float32")[tr_labels]
    y_va = np.eye(config.NUM_CLASSES, dtype="float32")[split["val"][1]]
    print(f"loaded {len(x_tr)} train / {len(x_va)} val images in {time.time() - t0:.0f}s")

    model, n_frozen_bn = build_v2_model(lr=args.lr)
    base = model.get_layer("efficientnetb3")
    head_layers = model.layers[1:]

    # ---- stage 1: head on cached features ------------------------------------
    head_path = run_dir / "head.weights.h5"
    if head_path.exists():
        model.load_weights(str(head_path))
        print(f"stage 1: reusing {head_path}")
    else:
        feat_path = run_dir / "features.npz"
        if feat_path.exists():
            f = np.load(feat_path)
            f_tr, f_va = f["train"], f["val"]
        else:
            t1 = time.time()
            base.trainable = False
            f_tr = np.concatenate([_features(base, x_tr), _features(base, x_tr[:, :, ::-1])])
            f_va = _features(base, x_va)
            np.savez(feat_path, train=f_tr, val=f_va)
            meta["feature_seconds"] = round(time.time() - t1)
            print(f"stage 1: cached features in {meta['feature_seconds']}s")
        head = tf.keras.Sequential([Input(f_tr.shape[1:])] + head_layers)
        head.compile(optimizer=Adamax(learning_rate=HEAD_LEARNING_RATE),
                     loss="categorical_crossentropy", metrics=["accuracy"])
        t1 = time.time()
        hh = head.fit(f_tr, np.concatenate([y_tr, y_tr]), validation_data=(f_va, y_va),
                      epochs=args.head_epochs, batch_size=32, shuffle=True, verbose=0,
                      callbacks=[CSVLogger(str(run_dir / "history_head.csv")),
                                 EarlyStopping(monitor="val_accuracy", mode="max",
                                               patience=8, restore_best_weights=True)])
        meta["stage1"] = {"seconds": round(time.time() - t1),
                          "epochs_run": len(hh.history["loss"]),
                          "best_val_accuracy": max(hh.history["val_accuracy"])}
        model.save_weights(str(head_path))
        print(f"stage 1: {meta['stage1']}")
        save_meta()

    # ---- stage 2: fine-tune the top of the backbone --------------------------
    n_trainable_layers = unfreeze_from(model, args.unfreeze_from)
    if not args.train_head_bn:
        for layer in head_layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = False
    frozen_bn_ok = all(not layer.trainable for layer in base.layers
                       if isinstance(layer, BatchNormalization))
    if not frozen_bn_ok:
        raise RuntimeError("a backbone BatchNorm layer is trainable")
    model.compile(optimizer=Adamax(learning_rate=args.lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    n_params = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"stage 2: {n_trainable_layers} backbone layers trainable from "
          f"{args.unfreeze_from}, {n_frozen_bn} backbone BN frozen, "
          f"{n_params:,} trainable params")
    meta["stage2_trainable_params"] = n_params

    train_ds = (tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
                .shuffle(len(x_tr), seed=config.SEED, reshuffle_each_iteration=True)
                .map(lambda x, y: (tf.cast(x, tf.float32), y))
                .batch(args.batch_size)
                .map(_augment_fn(), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = tf.data.Dataset.from_tensor_slices((x_va, y_va)) \
        .map(lambda x, y: (tf.cast(x, tf.float32), y)).batch(64)

    t1 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2,
        callbacks=[
            # Resumable: re-running with the same --run-name continues from the
            # last completed epoch, including optimizer state.
            BackupAndRestore(str(run_dir / "backup")),
            ModelCheckpoint(str(run_dir / "best.weights.h5"), monitor="val_accuracy",
                            mode="max", save_best_only=True, save_weights_only=True,
                            verbose=1),
            CSVLogger(str(run_dir / "history_finetune.csv"), append=True),
            ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.3,
                              patience=1, verbose=1),
            EarlyStopping(monitor="val_accuracy", mode="max", patience=3,
                          restore_best_weights=True, verbose=1),
        ],
    )
    probe = x_va[:32].astype("float32")
    model.load_weights(str(run_dir / "best.weights.h5"))
    meta["export_max_diff"] = export_for_serving(model, run_dir / "best_serving.h5", probe)
    model.load_weights(str(head_path))
    export_for_serving(model, run_dir / "head_serving.h5", probe)
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    meta.update({
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage2_seconds_this_session": round(time.time() - t1),
        "best_val_accuracy_this_session": max(hist["val_accuracy"]),
        "history_this_session": hist,
    })
    save_meta()
    print(f"best val_accuracy {max(hist['val_accuracy']):.4f} -> "
          f"{run_dir / 'best_serving.h5'}")


if __name__ == "__main__":
    main()
