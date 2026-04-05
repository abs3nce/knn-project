"""A module for defining local models."""

from abc import abstractmethod
from pathlib import Path

from n2f.models.model import Model


class LocalModel(Model):
    """A base class for local models."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

    @abstractmethod
    def save(self, save_path: Path) -> None:
        """Saves the model to the specified path."""

    @abstractmethod
    def train(self) -> None:
        """Trains the model."""
