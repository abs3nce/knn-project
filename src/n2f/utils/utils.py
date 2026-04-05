"""A module for utility functions."""

import re


def strip_markdown_json(text: str) -> str:
    """Strips markdown code block formatting from a JSON string."""
    code_block_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def format_error_message(error: Exception) -> str:
    """Formats an error message for logging."""
    return f"{type(error).__name__}: {str(error)}"
