"""A module for defining the response from the model."""

from dataclasses import dataclass


@dataclass
class Response:
    """Represents the response from the model."""

    text: str
    tokens_used: int
    success: bool
    error_message: str | None
