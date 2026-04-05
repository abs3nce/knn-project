"""A module for defining the base model class."""

from abc import ABC, abstractmethod
from pathlib import Path

from n2f.core.response import Response


class Model(ABC):
    """An abstract base class for models."""

    @abstractmethod
    def predict(
        self,
        prompt: str,
        image_paths: list[Path],
        max_tokens: int | None = None,
    ) -> Response:
        """Runs model inference based on the prompt and images."""
