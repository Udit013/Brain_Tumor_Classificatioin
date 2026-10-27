"""Staged pre-flight checks for the v2 recipe — run before any long training job.

Each check is cheap (the whole script takes ~2 minutes on an M3 Pro GPU) and
targets a way the first v2 run silently failed or could fail:

  1. label mapping     split labels agree with class-directory names
  2. forward pass      output shape, finite, softmax rows sum to 1
  3. frozen BatchNorm  78 backbone BN layers frozen; their moving statistics do
                       not move during a training step (inference mode)
  4. gradients         every trainable variable gets a finite, non-zero gradient
  5. augmentation      range [0, 255], shape kept, labels untouched, images changed
  6. overfit test      a 32-image subset must be memorised at lr 1e-4; lr 1e-3
                       is run alongside as the known-bad control
  7. graph == eager    predict (graph mode) matches an eager forward pass on the
                       device in use — catches the tensorflow-metal graph-mode
                       defect (config.disable_metal_graph_optimizer)

Usage:
    python scripts/v2_sanity_checks.py [--device {auto,cpu}] [--steps 40]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from btc import config  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    ok = bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--no-metal-fix", action="store_true",
                    help="skip the Metal graph workaround (to reproduce the defect)")
    args = ap.parse_args()

    import tensorflow as tf
    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    elif not args.no_metal_fix:
        config.disable_metal_graph_optimizer()
    from tensorflow.keras.layers import BatchNormalization

    from btc.preprocess import load_images
    from btc.splits import load_split
    from btc.train_v2 import _augment_fn, build_v2_model

    tf.keras.utils.set_random_seed(config.SEED)
    results: dict[str, bool] = {}
    t_start = time.time()

    # 1. label mapping ---------------------------------------------------------
    split = load_split()
    bad = [p for part in ("train", "val") for p, y in zip(*split[part])
           if Path(p).parent.name != config.CLASS_NAMES[y]]
    counts = {part: np.bincount(split[part][1], minlength=config.NUM_CLASSES).tolist()
              for part in ("train", "val")}
    results["label_mapping"] = check("label mapping", not bad,
                                     f"{len(bad)} mismatches; per-class counts {counts}")

    # a small balanced subset: 8 per class from the training split
    rng = np.random.default_rng(config.SEED)
    paths, labels = np.array(split["train"][0]), np.array(split["train"][1])
    idx = np.concatenate([rng.choice(np.flatnonzero(labels == c), 8, replace=False)
                          for c in range(config.NUM_CLASSES)])
    x = load_images(list(paths[idx]))
    y = tf.one_hot(labels[idx], config.NUM_CLASSES)

    # 2. forward pass ----------------------------------------------------------
    model, n_frozen = build_v2_model()
    p = model.predict(x, batch_size=16, verbose=0)
    results["forward"] = check(
        "forward pass", p.shape == (32, 4) and np.isfinite(p).all()
        and np.allclose(p.sum(1), 1, atol=1e-4),
        f"shape {p.shape}, row sums {p.sum(1).min():.5f}..{p.sum(1).max():.5f}")

    # 7. graph == eager (run on the freshly built model, before any training) --
    eager = np.concatenate([model(tf.constant(x[i:i + 16]), training=False).numpy()
                            for i in range(0, len(x), 16)])
    gdiff = float(np.abs(eager - p).max())
    results["graph_eager"] = check("graph mode == eager", gdiff < 1e-4,
                                   f"max |predict - eager| {gdiff:.2e}")

    # 3. frozen BatchNorm --------------------------------------------------------
    base = model.get_layer("efficientnetb3")
    bns = [layer for layer in base.layers if isinstance(layer, BatchNormalization)]
    before = [layer.moving_mean.numpy().copy() for layer in bns]
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        loss = loss_fn(y[:16], model(x[:16], training=True))
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    moved = max(float(np.abs(layer.moving_mean.numpy() - b).max())
                for layer, b in zip(bns, before))
    head_bns = [layer for layer in model.layers if isinstance(layer, BatchNormalization)]
    results["frozen_bn"] = check(
        "frozen backbone BatchNorm", n_frozen == len(bns) == 78 and moved == 0.0,
        f"{n_frozen}/{len(bns)} frozen, max moving-mean change {moved:.2e}"
        + f", head BN trainable={[layer.trainable for layer in head_bns]}")

    # 4. gradients ---------------------------------------------------------------
    norms = [float(tf.norm(g)) if g is not None else None for g in grads]
    missing = sum(n is None for n in norms)
    nonfinite = sum(n is not None and not np.isfinite(n) for n in norms)
    zero = sum(n == 0.0 for n in norms if n is not None)
    results["gradients"] = check(
        "finite gradients", missing == 0 and nonfinite == 0 and zero == 0
        and np.isfinite(float(loss)),
        f"{len(norms)} trainable vars, {missing} None, {nonfinite} non-finite, "
        f"{zero} zero; loss {float(loss):.3f}")

    # 5. augmentation ------------------------------------------------------------
    xa, ya = _augment_fn()(tf.constant(x[:16]), y[:16])
    xa = xa.numpy()
    results["augmentation"] = check(
        "augmentation", xa.shape == x[:16].shape and xa.min() >= 0 and xa.max() <= 255
        and np.array_equal(ya.numpy(), y[:16].numpy())
        and float(np.abs(xa - x[:16]).mean()) > 1.0,
        f"range [{xa.min():.1f}, {xa.max():.1f}], mean |change| "
        f"{float(np.abs(xa - x[:16]).mean()):.1f}")

    # 6. overfit test ------------------------------------------------------------
    overfit = {}
    for lr in (1e-4, 1e-3):
        tf.keras.utils.set_random_seed(config.SEED)
        m, _ = build_v2_model(lr=lr)
        t0 = time.time()
        h = m.fit(x, y, batch_size=16, epochs=args.steps // 2, shuffle=True, verbose=0)
        acc = float(m.evaluate(x, y, batch_size=16, verbose=0)[1])
        overfit[lr] = {"final_train_loss": round(float(h.history["loss"][-1]), 4),
                       "subset_accuracy_eval_mode": round(acc, 4),
                       "seconds": round(time.time() - t0, 1)}
        print(f"       lr {lr:g}: {overfit[lr]}")
    results["overfit"] = check(
        "overfit 32 images at lr 1e-4", overfit[1e-4]["subset_accuracy_eval_mode"] >= 0.9,
        f"lr 1e-4 → {overfit[1e-4]['subset_accuracy_eval_mode']:.0%}, "
        f"lr 1e-3 (control) → {overfit[1e-3]['subset_accuracy_eval_mode']:.0%}")

    gpus = tf.config.get_visible_devices("GPU")
    summary = {"device": "gpu" if gpus else "cpu", "checks": results,
               "overfit": {str(k): v for k, v in overfit.items()},
               "seconds": round(time.time() - t_start)}
    print(json.dumps(summary, indent=2))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
