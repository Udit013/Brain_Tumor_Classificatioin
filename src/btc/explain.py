"""Module 5 — Grad-CAM explainability.

Generates Grad-CAM heatmap overlays for the EfficientNetB3 classifier and saves
a montage covering, for every one of the 4 classes, both a CORRECT and an
INCORRECT (where available) test prediction — so the reviewer can see what the
network attends to when it is right vs. wrong.

Outputs:
  results/figures/gradcam/<class>_<correct|wrong>_<i>.png   (individual overlays)
  results/figures/gradcam_montage.png                       (4x2 summary grid)

Usage:
    python -m btc.explain                 # auto-select examples from test set
    python -m btc.explain --image PATH    # single-image overlay (also used by API)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import config


def _grad_model(model, last_conv_name):
    """Return (base_grad_model, head_layers) for Grad-CAM on the nested model.

    A single Model spanning the outer input and an inner conv tensor triggers a
    "Graph disconnected" error because the backbone is a nested Functional model.
    Instead we build a grad model over the BACKBONE alone (its own connected
    graph), mapping base.input -> [last_conv_output, base_output], and apply the
    classifier head functionally on top. The returned tuple is consumed by
    ``gradcam_heatmap``.
    """
    import tensorflow as tf

    base = model.get_layer("efficientnetb3")
    conv_layer = base.get_layer(last_conv_name)
    base_grad = tf.keras.models.Model(
        inputs=base.input,
        outputs=[conv_layer.output, base.output],
    )
    head_layers = model.layers[1:]  # everything after the backbone
    return base_grad, head_layers


def gradcam_heatmap(x_single, model, grad_model=None, class_idx=None):
    """Return a HxW heatmap in [0,1] for one image (shape (256,256,3), [0,255])."""
    import tensorflow as tf

    if grad_model is None:
        from .model import find_last_conv_layer_name
        grad_model = _grad_model(model, find_last_conv_layer_name(model))
    base_grad, head_layers = grad_model

    arr = tf.convert_to_tensor(x_single[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, base_out = base_grad(arr)
        tape.watch(conv_out)
        h = base_out
        for layer in head_layers:
            h = layer(h, training=False)
        preds = h
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))           # channel weights
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)       # weighted sum
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-9)
    return heatmap.numpy(), class_idx


def overlay(x_single, heatmap, alpha=0.4):
    """Blend heatmap (HxW [0,1]) over the image (256,256,3 [0,255]) -> uint8 RGB."""
    import cv2

    hm = cv2.resize(heatmap, (x_single.shape[1], x_single.shape[0]))
    hm = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    base = np.uint8(np.clip(x_single, 0, 255))
    return np.uint8(base * (1 - alpha) + hm_color * alpha)


def explain_image(image_path, model=None):
    """Grad-CAM for a single arbitrary image. Returns (overlay_rgb, class_idx, probs)."""
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from .model import load_trained_model, find_last_conv_layer_name

    model = model or load_trained_model()
    grad_model = _grad_model(model, find_last_conv_layer_name(model))
    img = load_img(str(image_path), target_size=config.IMG_SIZE, color_mode="rgb")
    x = img_to_array(img).astype("float32")
    probs = model.predict(x[None, ...], verbose=0)[0]
    heatmap, class_idx = gradcam_heatmap(x, model, grad_model,
                                         class_idx=int(np.argmax(probs)))
    return overlay(x, heatmap), int(class_idx), probs


def _montage(examples, path):
    import matplotlib.pyplot as plt

    n = len(examples)
    fig, axes = plt.subplots(n, 1, figsize=(5, 4 * n))
    if n == 1:
        axes = [axes]
    for ax, (title, ov) in zip(axes, examples):
        ax.imshow(ov)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM overlays")
    parser.add_argument("--image", type=str, default=None)
    args = parser.parse_args()

    config.ensure_dirs()
    out_dir = config.FIGURES_DIR / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    from .model import load_trained_model
    import imageio.v2 as imageio

    model = load_trained_model()

    if args.image:
        ov, cls, probs = explain_image(args.image, model)
        dest = out_dir / (Path(args.image).stem + "_gradcam.png")
        imageio.imwrite(dest, ov)
        print(f"Predicted: {config.CLASS_NAMES[cls]} ({probs[cls]:.4f}) -> {dest}")
        return

    # Auto: one correct + one wrong per class from the test set.
    from .data import load_test_arrays
    from .model import find_last_conv_layer_name

    x, y_true, _ = load_test_arrays()
    probs = model.predict(x, batch_size=config.BATCH_SIZE, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    grad_model = _grad_model(model, find_last_conv_layer_name(model))

    montage = []
    for ci, name in enumerate(config.CLASS_NAMES):
        for kind, mask in (("correct", (y_true == ci) & (y_pred == ci)),
                           ("wrong", (y_true == ci) & (y_pred != ci))):
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            i = int(idxs[0])
            hm, _ = gradcam_heatmap(x[i], model, grad_model, class_idx=int(y_pred[i]))
            ov = overlay(x[i], hm)
            title = (f"true={name} pred={config.CLASS_NAMES[y_pred[i]]} "
                     f"({probs[i, y_pred[i]]:.2f}) [{kind}]")
            imageio.imwrite(out_dir / f"{name}_{kind}_{i}.png", ov)
            montage.append((title, ov))

    _montage(montage, config.FIGURES_DIR / "gradcam_montage.png")
    print(f"Saved {len(montage)} overlays -> {out_dir}")
    print(f"Saved montage -> {config.FIGURES_DIR / 'gradcam_montage.png'}")


if __name__ == "__main__":
    main()
