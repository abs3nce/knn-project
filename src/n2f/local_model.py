from abc import abstractmethod
from pathlib import Path

from n2f.model import Model


class LocalModel(Model):
    @abstractmethod
    def save(self, save_path: Path) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass
