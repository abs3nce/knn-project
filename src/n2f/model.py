from abc import ABC, abstractmethod
from pathlib import Path


class Model(ABC):
    @abstractmethod
    def predict(
        self,
        prompt: str,
        image_paths: list[Path],
        max_tokens: int = 2048,
    ) -> str:
        pass
