"""Recompute BatchNorm moving statistics, then re-evaluate (full test set).

Legitimate fix for inference-time BN collapse: run forward passes over TRAINING
data with training=True so every BN layer's moving_mean / moving_variance are
re-estimated to match the data — WITHOUT any optimizer step, so learned weights
are unchanged. Then evaluate in normal inference mode.
"""
import numpy as np
import tensorflow as tf
from btc.model import load_trained_model
from btc.data import make_generators, load_test_arrays
from btc import config

model = load_trained_model()

# Lower BN momentum so stats converge within one pass, and reset accumulators.
import tensorflow.keras.layers as L


def _reset_bn(m):
    n = 0
    for layer in getattr(m, "layers", []):
        if isinstance(layer, L.BatchNormalization):
            layer.momentum = 0.9
            n += 1
        if hasattr(layer, "layers"):
            n += _reset_bn(layer)
    return n


print("BN layers retuned:", _reset_bn(model))

train_gen, _ = make_generators()
steps = 350  # ~ full training set at batch 16
for i in range(steps):
    xb, _ = next(train_gen)
    model(xb, training=True)  # updates BN moving stats only
    if (i + 1) % 100 == 0:
        print(f"  recalibrated {i+1}/{steps} batches")

x, y, _ = load_test_arrays()
p = model.predict(x, batch_size=16, verbose=0)
acc = (p.argmax(1) == y).mean()
print(f"\nFULL-TEST accuracy after BN recalibration: {acc:.5f}")
print("pred dist:", np.bincount(p.argmax(1), minlength=4), "(true: 400 each)")

# Save the recalibrated weights to a separate file (do not clobber the parity run).
out = config.MODELS_DIR / "EfficientNetB3_recalibrated.h5"
model.save_weights(str(out))
print(f"saved -> {out}")
