"""Module 1 — Evaluation rigor on the published test split.

Computes and saves (all REAL, measured at run time):
  * overall accuracy (confirms the published 99.844% headline)
  * per-class precision / recall / F1 + support (classification_report)
  * confusion matrix (figure + raw counts)
  * per-class one-vs-rest ROC curves + AUC
  * per-class precision-recall curves + average precision

Outputs:
  results/metrics/evaluation.json
  results/figures/confusion_matrix.png
  results/figures/roc_curves.png
  results/figures/pr_curves.png

Usage:
    python -m btc.evaluate
"""

from __future__ import annotations

import json

import numpy as np

from . import config


def predict_test(model=None):
    """Return (y_true, y_prob) over the ordered test set.

    y_true: (N,) int labels; y_prob: (N, num_classes) softmax probabilities.
    """
    from .model import load_trained_model
    from .data import load_test_arrays

    model = model or load_trained_model()
    x, y_true, _ = load_test_arrays()
    y_prob = model.predict(x, batch_size=config.BATCH_SIZE, verbose=0)
    return y_true, y_prob


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
        average_precision_score,
    )
    from sklearn.preprocessing import label_binarize

    y_pred = np.argmax(y_prob, axis=1)
    labels = list(range(config.NUM_CLASSES))
    y_onehot = label_binarize(y_true, classes=labels)

    report = classification_report(
        y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True, digits=5
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class_auc, per_class_ap = {}, {}
    for i, name in enumerate(config.CLASS_NAMES):
        per_class_auc[name] = float(roc_auc_score(y_onehot[:, i], y_prob[:, i]))
        per_class_ap[name] = float(average_precision_score(y_onehot[:, i], y_prob[:, i]))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_roc_auc": float(np.mean(list(per_class_auc.values()))),
        "macro_avg_precision": float(np.mean(list(per_class_ap.values()))),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "per_class_roc_auc": per_class_auc,
        "per_class_avg_precision": per_class_ap,
        "n_test": int(len(y_true)),
        "published_reference_accuracy": config.PUBLISHED_ACCURACY[config.PRIMARY_MODEL],
    }


def _plot_confusion_matrix(cm: np.ndarray, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(config.NUM_CLASSES),
        yticks=np.arange(config.NUM_CLASSES),
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (published test split)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_roc(y_true, y_prob, path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_onehot = label_binarize(y_true, classes=list(range(config.NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(config.CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="Per-class ROC (one-vs-rest)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pr(y_true, y_prob, path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize

    y_onehot = label_binarize(y_true, classes=list(range(config.NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(config.CLASS_NAMES):
        prec, rec, _ = precision_recall_curve(y_onehot[:, i], y_prob[:, i])
        ap = average_precision_score(y_onehot[:, i], y_prob[:, i])
        ax.plot(rec, prec, label=f"{name} (AP={ap:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Per-class Precision-Recall")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    config.ensure_dirs()
    y_true, y_prob = predict_test()
    metrics = compute_metrics(y_true, y_prob)

    out = config.METRICS_DIR / "evaluation.json"
    out.write_text(json.dumps(metrics, indent=2))

    _plot_confusion_matrix(np.array(metrics["confusion_matrix"]),
                           config.FIGURES_DIR / "confusion_matrix.png")
    _plot_roc(y_true, y_prob, config.FIGURES_DIR / "roc_curves.png")
    _plot_pr(y_true, y_prob, config.FIGURES_DIR / "pr_curves.png")

    print(f"Accuracy (reproduced): {metrics['accuracy']:.5f}")
    print(f"Published reference   : {metrics['published_reference_accuracy']:.5f}")
    print(f"Macro ROC-AUC         : {metrics['macro_roc_auc']:.5f}")
    print(f"Saved metrics -> {out}")
    print(f"Saved figures -> {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
