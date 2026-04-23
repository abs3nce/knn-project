"""A module for defining prompts used in the annotation process."""

from abc import ABC, abstractmethod
from pathlib import Path

from jinja2 import Template


class Prompt(ABC):
    """Abstract base class for prompts used in the annotation process."""

    def __init__(self, template_file_path: Path) -> None:
        self.path = template_file_path

    @abstractmethod
    def render(self, arguments: dict[str, str]) -> str:
        """Renders the prompt template and returns the resulting string."""


class AnnotatePrompt(Prompt):
    """A prompt for annotating images."""

    def __init__(self, template_file_path: Path) -> None:
        super().__init__(template_file_path)
        template_content = template_file_path.read_text(encoding="utf-8")
        self.template = Template(template_content)

    def render(self, arguments: dict[str, str]) -> str:
        return self.template.render(**arguments)
