from abc import ABC, abstractmethod
from pathlib import Path

from n2f.response import Response


class Model(ABC):
    @abstractmethod
    def predict(
        self,
        prompt: str,
        image_paths: list[Path],
        max_tokens: int | None = None,
    ) -> Response:
        pass
