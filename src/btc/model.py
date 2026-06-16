"""EfficientNetB3 model definition — exact parity with the published notebook.

Reproduces, layer-for-layer, the architecture in
``legacy/EfficientNetB3_Brain_Tumor.ipynb`` (cell 14):

    EfficientNetB3(include_top=False, weights='imagenet',
                   input_shape=(256,256,3), pooling='max')
      -> BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)
      -> Dense(512, l2(0.016) kernel, l1(0.006) activity, l1(0.006) bias, relu)
      -> Dropout(0.4, seed=75)
      -> Dense(256, same regularisers, relu)
      -> Dropout(0.2, seed=75)
      -> Dense(4, softmax)
    compile(Adamax(1e-3), categorical_crossentropy, ['accuracy'])

Total params ~11.7M (matches the paper's "lowest of the four").
"""

from __future__ import annotations

from . import config


def build_model(weights: str | None = "imagenet"):
    """Build the published EfficientNetB3 classifier.

    Parameters
    ----------
    weights:
        Passed to the EfficientNetB3 backbone. Use ``"imagenet"`` for training
        (default, parity) or ``None`` when you will immediately load fine-tuned
        weights from disk and want to skip the imagenet download.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adamax
    from tensorflow.keras import regularizers

    base_model = tf.keras.applications.efficientnet.EfficientNetB3(
        include_top=False,
        weights=weights,
        input_shape=config.IMG_SHAPE,
        pooling="max",
    )

    model = Sequential(
        [
            base_model,
            BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
            Dense(
                512,
                kernel_regularizer=regularizers.l2(config.L2_KERNEL),
                activity_regularizer=regularizers.l1(config.L1_ACTIVITY),
                bias_regularizer=regularizers.l1(config.L1_BIAS),
                activation="relu",
            ),
            Dropout(rate=0.4, seed=config.DROPOUT_SEED),
            Dense(
                256,
                kernel_regularizer=regularizers.l2(config.L2_KERNEL),
                activity_regularizer=regularizers.l1(config.L1_ACTIVITY),
                bias_regularizer=regularizers.l1(config.L1_BIAS),
                activation="relu",
            ),
            Dropout(rate=0.2, seed=config.DROPOUT_SEED),
            Dense(config.NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adamax(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_trained_model(weights_path=None):
    """Build the architecture and load fine-tuned weights from disk.

    Raises a clear error (rather than silently using imagenet-only weights) if
    the weights file is missing, so downstream evaluation never reports numbers
    from an untrained head.
    """
    weights_path = weights_path or config.WEIGHTS_PATH
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained weights not found at {weights_path}.\n"
            "Run `python -m btc.train` (or scripts/reproduce.sh) first."
        )
    # weights='imagenet' keeps the backbone init identical before load; using
    # None would also work since we overwrite all weights immediately.
    model = build_model(weights=None)
    model.load_weights(str(weights_path))
    return model


def find_last_conv_layer_name(model) -> str:
    """Return the name of the last 4-D conv feature map (for Grad-CAM).

    EfficientNetB3's final conv activation is ``top_activation``; we locate it
    robustly inside the nested functional ``efficientnetb3`` submodel.
    """
    base = model.get_layer("efficientnetb3")
    for layer in reversed(base.layers):
        if len(getattr(layer, "output_shape", ())) == 4:
            return layer.name
    raise RuntimeError("No 4-D conv layer found in EfficientNetB3 backbone.")
