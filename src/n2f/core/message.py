"""A module for representing messages exchanged with a model."""

from typing import TypedDict


class Message(TypedDict):
    """Represents a message with a role and content."""

    role: str
    content: str | list[dict[str, str]]
