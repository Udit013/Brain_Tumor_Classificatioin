"""Closed-form classifier head and Grad-CAM, in NumPy (no TensorFlow).

The deployed model is EfficientNetB3 -> top_activation (8x8x1536) -> global
max-pool -> BatchNorm -> Dense(512, relu) -> Dense(256, relu) -> Dense(4,
softmax). Given the conv map from the ONNX serving graph and the head weights
(head.npz, written by btc.export_serving), this module reproduces

  * the softmax probabilities, and
  * Grad-CAM exactly as btc.explain computes it: gradient of the class
    probability with respect to the conv map, channel weights = spatial mean
    of that gradient, heatmap = relu(sum_c weight_c * conv_c), max-normalised.

TensorFlow's gradient of reduce_max splits the incoming gradient equally among
tied maxima; `gradcam_from_conv` does the same. Dropout is the identity at
inference, and the head's activity regularisers only add training losses.
"""

from __future__ import annotations

import numpy as np


def head_forward(pooled: np.ndarray, head: dict) -> tuple[np.ndarray, dict]:
    """pooled (1536,) -> (softmax probs (4,), intermediates for the backward pass)."""
    f = pooled.astype("float64")
    scale = head["bn_gamma"] / np.sqrt(head["bn_var"] + head["bn_epsilon"])
    z0 = (f - head["bn_mean"]) * scale + head["bn_beta"]
    u1 = z0 @ head["w1"] + head["b1"]
    a1 = np.maximum(u1, 0.0)
    u2 = a1 @ head["w2"] + head["b2"]
    a2 = np.maximum(u2, 0.0)
    logits = a2 @ head["w3"] + head["b3"]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    return probs, {"scale": scale, "u1": u1, "u2": u2}


def class_score_gradient(pooled: np.ndarray, head: dict, class_idx: int) -> np.ndarray:
    """d probs[class_idx] / d pooled — shape (1536,)."""
    probs, mid = head_forward(pooled, head)
    onehot = np.zeros_like(probs)
    onehot[class_idx] = 1.0
    d_logits = probs[class_idx] * (onehot - probs)            # softmax Jacobian row
    d_a2 = d_logits @ head["w3"].T
    d_u2 = d_a2 * (mid["u2"] > 0)
    d_a1 = d_u2 @ head["w2"].T
    d_u1 = d_a1 * (mid["u1"] > 0)
    d_z0 = d_u1 @ head["w1"].T
    return d_z0 * mid["scale"]


def gradcam_from_conv(conv: np.ndarray, head: dict, class_idx: int) -> np.ndarray:
    """conv (H, W, C) float32 -> Grad-CAM heatmap (H, W) in [0, 1]."""
    conv32 = conv.astype("float32")
    pooled = conv32.max(axis=(0, 1))
    is_max = conv32 == pooled                                  # ties share the gradient
    count = is_max.sum(axis=(0, 1))
    g_pooled = class_score_gradient(pooled, head, class_idx)
    grads = is_max * (g_pooled / count)                        # (H, W, C)
    weights = grads.mean(axis=(0, 1))
    cam = np.maximum((conv32.astype("float64") * weights).sum(axis=-1), 0.0)
    return (cam / (cam.max() + 1e-9)).astype("float32")
