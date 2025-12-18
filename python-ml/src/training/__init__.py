"""Training pipelines for sentiment models."""

from .data_loader import load_labeled_data, rating_to_label
from .setfit_trainer import train_setfit

__all__ = [
    "load_labeled_data",
    "rating_to_label",
    "train_setfit",
]
