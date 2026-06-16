"""Decisive diagnostic for the accuracy-collapse anomaly.

Compares, on a balanced sample of the test set:
  (a) inference-mode prediction  (model.predict / training=False) -> BN moving avg
  (b) training-mode prediction   (model(x, training=True))        -> BN batch stats

If (b) is much better than (a), the EfficientNet BatchNorm moving-average
statistics are the culprit (the classic high-momentum/few-steps instability),
not a label or preprocessing bug.
"""
import numpy as np
from btc.model import load_trained_model
from btc.data import load_test_arrays
from btc import config

x, y, _ = load_test_arrays()
# balanced 80/class sample for speed
idx = np.concatenate([np.where(y == c)[0][:80] for c in range(config.NUM_CLASSES)])
xs, ys = x[idx], y[idx]

model = load_trained_model()

p_inf = model.predict(xs, batch_size=16, verbose=0)
acc_inf = (p_inf.argmax(1) == ys).mean()

# training=True -> BN uses batch statistics
import tensorflow as tf
p_tr = model(tf.convert_to_tensor(xs), training=True).numpy()
acc_tr = (p_tr.argmax(1) == ys).mean()

print(f"inference-mode (BN moving avg) accuracy: {acc_inf:.4f}")
print(f"training-mode  (BN batch stats) accuracy: {acc_tr:.4f}")
print("pred dist (inference):", np.bincount(p_inf.argmax(1), minlength=4))
print("pred dist (training): ", np.bincount(p_tr.argmax(1), minlength=4))
