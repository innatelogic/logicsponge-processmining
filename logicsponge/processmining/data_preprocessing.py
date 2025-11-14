"""
Deprecated: windowed dataset generation has been removed.

Windowing is now applied solely by slicing inputs inside training/evaluation
functions. This module remains only to avoid import errors in legacy code.
"""

from __future__ import annotations

import warnings


def __getattr__(name: str):  # pragma: no cover - compatibility shim
    warnings.warn(
        "data_preprocessing.* is deprecated; windowing is applied by slicing in"
        " train/evaluate functions. Remove any dependency on these utilities.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise AttributeError(
        "data_preprocessing no longer provides windowed dataset generation."
    )
