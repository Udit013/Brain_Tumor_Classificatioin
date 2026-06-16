"""Brain Tumor Classification — production extension package.

This package wraps the published IEEE 2024 EfficientNetB3 brain-tumor
classifier (see ../legacy for the original, byte-for-byte notebooks) with a
production-grade evaluation, calibration, explainability, serving and
monitoring stack.

The model definition, preprocessing and training schedule in :mod:`btc.model`
and :mod:`btc.data` reproduce the published EfficientNetB3 pipeline exactly so
that the headline 99.844% test accuracy can be confirmed before any extension
is layered on top.
"""

__version__ = "0.1.0"
