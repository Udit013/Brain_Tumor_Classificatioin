"""Module 4 — Confidence calibration.

Reports Expected Calibration Error (ECE) and a reliability diagram BEFORE and
AFTER temperature scaling, and persists the fitted temperature so the serving
layer can return calibrated confidences.

METHOD
------
The published head ends in softmax, so we recover logits as ``log(prob)``
(softmax is invariant to an additive constant, so this is a valid logit up to a
constant and does not change temperature-scaling results). Temperature ``T`` is
fitted by minimising negative log-likelihood: ``p = softmax(log(prob)/T)``.

PROTOCOL
--------
By default T is fit and ECE reported on the test split (the protocol behind
the reported numbers); a stricter 5-seed fit-on-one-half/report-on-the-other
check gave essentially the same ECE (0.074 +/- 0.008). --holdout fits on a
disjoint slice of the test set. --fit-on val fits on the leakage-safe
validation split (btc.splits) — correct for a model that never trained on it,
but NOT for the parity model, whose training data includes those images.

Outputs:
  results/metrics/calibration.json
  results/figures/reliability_diagram.png
  models/temperature.json

Usage:
    python -m btc.calibration
"""

from __future__ import annotations

import json

import numpy as np

from . import config

N_BINS = 15


def expected_calibration_error(probs, y_true, n_bins=N_BINS):
    """Standard top-label ECE."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    bin_stats = []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        # Half-open bins (lo, hi], except the first, which is closed [lo, hi]
        # so a confidence of exactly 0.0 is still counted. Without this the
        # first bin is exclusive at 0.0 and such samples silently vanish from
        # the ECE weighting. (Unreachable for a 4-class softmax, where
        # confidence >= 0.25, but correct for any general use of this helper.)
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if mask.any():
            acc = correct[mask].mean()
            avg_conf = conf[mask].mean()
            w = mask.mean()
            ece += w * abs(acc - avg_conf)
            bin_stats.append({"lo": float(lo), "hi": float(hi),
                              "acc": float(acc), "conf": float(avg_conf),
                              "count": int(mask.sum())})
    return float(ece), bin_stats


def _fit_temperature(probs, y_true):
    """Fit scalar T>0 minimising NLL of softmax(log(probs)/T) via grid+refine."""
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    onehot = np.eye(config.NUM_CLASSES)[y_true]

    def nll(T):
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(axis=1, keepdims=True)
        return float(-(onehot * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1).mean())

    # coarse grid then local refine
    grid = np.linspace(0.05, 5.0, 100)
    best_T = min(grid, key=nll)
    fine = np.linspace(max(0.05, best_T - 0.1), best_T + 0.1, 100)
    best_T = float(min(fine, key=nll))
    return best_T


def apply_temperature(probs, T):
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / T
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def _plot_reliability(probs_pre, probs_post, y_true, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, probs, title in (
        (axes[0], probs_pre, "Before (uncalibrated)"),
        (axes[1], probs_post, "After temperature scaling"),
    ):
        ece, stats = expected_calibration_error(probs, y_true)
        confs = [s["conf"] for s in stats]
        accs = [s["acc"] for s in stats]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
        ax.bar(confs, accs, width=0.05, alpha=0.7, edgecolor="black",
               label="observed")
        ax.set(xlabel="Confidence", ylabel="Accuracy", xlim=(0, 1), ylim=(0, 1),
               title=f"{title}\nECE = {ece:.4f}")
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Calibration / temperature scaling")
    parser.add_argument("--fit-on", choices=["test", "val"], default="test",
                        help="test (default, the protocol behind the reported "
                             "numbers): fit on the test set, optionally a --holdout "
                             "slice of it. val: fit on the leakage-safe validation "
                             "split (btc.splits) and report on the full test set — "
                             "only valid for a model that never trained on those "
                             "images, so NOT for the parity model.")
    parser.add_argument("--holdout", type=float, default=0.0,
                        help="with --fit-on test: fraction of the test set used ONLY "
                             "for fitting T (the rest reports ECE). 0 = fit & report "
                             "on the full test split.")
    args = parser.parse_args()

    config.ensure_dirs()
    from .evaluate import predict_test

    y_true, probs = predict_test()
    rep_idx = np.arange(len(y_true))

    if args.fit_on == "val":
        # Clean protocol: the validation split (btc.splits) was never trained on
        # by the deployed model and shares no near-duplicates with its training
        # data, and the test set plays no part in choosing T.
        from .model import load_trained_model
        from .preprocess import load_images
        from .splits import load_split

        val_paths, y_fit = load_split()["val"]
        probs_fit = load_trained_model().predict(load_images(val_paths),
                                                 batch_size=config.BATCH_SIZE, verbose=0)
        protocol, caveat = "fit_on_val_report_on_test", (
            "T fit on the leakage-safe validation split; ECE reported on the full "
            "test set, which played no part in fitting.")
    elif args.holdout > 0:
        rng = np.random.default_rng(config.SEED)
        idx = rng.permutation(len(y_true))
        cut = int(len(y_true) * args.holdout)
        fit_idx, rep_idx = idx[:cut], idx[cut:]
        probs_fit, y_fit = probs[fit_idx], y_true[fit_idx]
        protocol, caveat = "test_holdout", "T fit on a disjoint holdout slice of the test split."
    else:
        probs_fit, y_fit = probs, y_true
        protocol, caveat = "fit_and_report_on_test", (
            "T fit and ECE reported on the same test split — can be optimistic.")

    T = _fit_temperature(probs_fit, y_fit)

    probs_post = apply_temperature(probs, T)
    ece_pre, _ = expected_calibration_error(probs[rep_idx], y_true[rep_idx])
    ece_post, _ = expected_calibration_error(probs_post[rep_idx], y_true[rep_idx])

    result = {
        "temperature": T,
        "ece_before": ece_pre,
        "ece_after": ece_post,
        "n_bins": N_BINS,
        "fit_protocol": protocol,
        "n_fit": int(len(y_fit)),
        "n_report": int(len(rep_idx)),
        "caveat": caveat,
    }
    (config.METRICS_DIR / "calibration.json").write_text(json.dumps(result, indent=2))
    config.TEMPERATURE_PATH.write_text(json.dumps({"temperature": T}, indent=2))
    _plot_reliability(probs[rep_idx], probs_post[rep_idx], y_true[rep_idx],
                      config.FIGURES_DIR / "reliability_diagram.png")

    print(json.dumps(result, indent=2))
    print(f"\nSaved temperature -> {config.TEMPERATURE_PATH}")


if __name__ == "__main__":
    main()
