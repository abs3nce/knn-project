"""A module for defining the model identifier."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelIdentifier:
    """A class for identifying models."""

    environment_category: str
    registry_key: str
    target_value: str

    def __str__(self) -> str:
        return f"{self.environment_category}:{self.registry_key}:{self.target_value}"
