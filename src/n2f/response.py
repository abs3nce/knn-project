from dataclasses import dataclass


@dataclass
class Response:
    text: str
    tokens_used: int
    success: bool
    error_message: str | None
