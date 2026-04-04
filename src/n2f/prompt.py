from abc import ABC, abstractmethod
from pathlib import Path

from jinja2 import Template


class Prompt(ABC):
    def __init__(self, template_file_path: Path) -> None:
        self.path = template_file_path

    @abstractmethod
    def render(self) -> str:
        pass


class AnnotatePrompt(Prompt):
    def __init__(self, template_file_path: Path) -> None:
        super().__init__(template_file_path)
        template_content = template_file_path.read_text(encoding="utf-8")
        self.template = Template(template_content)

    def render(self) -> str:
        return self.template.render()
