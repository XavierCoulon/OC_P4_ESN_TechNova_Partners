"""
TechNova Partners Employee Turnover Prediction Package

This package contains encoding utilities for categorical feature preprocessing
in machine learning pipelines.
"""

from .utils import (
    apply_binary_encoding,
    apply_label_encoding,
    apply_onehot_encoding,
    apply_ordinal_encoding,
)

__version__ = "1.0.0"
__author__ = "Data Science Team"

__all__ = [
    "apply_binary_encoding",
    "apply_label_encoding",
    "apply_onehot_encoding",
    "apply_ordinal_encoding",
]
