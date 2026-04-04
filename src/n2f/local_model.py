from abc import abstractmethod
from pathlib import Path

from n2f.model import Model


class LocalModel(Model):
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

    @abstractmethod
    def save(self, save_path: Path) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass
