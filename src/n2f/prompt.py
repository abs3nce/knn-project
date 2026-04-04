from abc import ABC, abstractmethod
from enum import StrEnum

from jinja2 import Environment, FileSystemLoader

template_environment = Environment(loader=FileSystemLoader("prompts/"))


class PromptTemplate(StrEnum):
    ANNOTATE_PROMPT = "annotate_prompt.j2"


class Prompt(ABC):
    @abstractmethod
    def render(self) -> str:
        pass


class AnnotatePrompt(Prompt):
    def __init__(self) -> None:
        self.template = template_environment.get_template(
            PromptTemplate.ANNOTATE_PROMPT.value
        )

    def render(self) -> str:
        return self.template.render()
