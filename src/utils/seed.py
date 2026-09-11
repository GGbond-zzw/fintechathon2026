"""Reproducible random seed setup."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy when available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)

