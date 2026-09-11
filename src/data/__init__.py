"""Data loading and validation interfaces."""

from .loader import load_test, load_train, load_train_tail
from .validator import validate_frame

__all__ = ["load_test", "load_train", "load_train_tail", "validate_frame"]

