"""A module for defining prompts used in the annotation process."""

from abc import ABC, abstractmethod
from pathlib import Path
import json
import re
from typing import TypedDict

from jinja2 import Template

from n2f.core.bounding_box import BoundingBox


class FewShotExample(TypedDict):
    image_path: Path
    label: str
    bbox_2d: list[int]


class Prompt(ABC):
    """Abstract base class for prompts used in the annotation process."""

    def __init__(self, template_file_path: Path) -> None:
        self.path = template_file_path

    @abstractmethod
    def render(self, arguments: dict[str, str]) -> str:
        """Renders the prompt template and returns the resulting string."""

    def image_paths(self) -> list[Path]:
        """Returns additional image paths that should be sent to the model."""
        return []


class AnnotatePrompt(Prompt):
    """A prompt for annotating images."""

    def __init__(self, template_file_path: Path) -> None:
        super().__init__(template_file_path)
        template_content = template_file_path.read_text(encoding="utf-8")
        self.template = Template(template_content)

    def render(self, arguments: dict[str, str]) -> str:
        return self.template.render(**arguments)


class FewShotAnnotatePrompt(Prompt):
    """A few-shot prompt with examples extracted from the template itself."""

    def __init__(self, template_file_path: Path) -> None:
        super().__init__(template_file_path)
        template_content = template_file_path.read_text(encoding="utf-8")
        self.template = Template(template_content)
        self._example_image_paths = self._extract_image_paths(template_content)
        self._examples: list[FewShotExample] = [
            self._load_example_data(image_path)
            for image_path in self._example_image_paths
        ]

    def render(self, arguments: dict[str, str]) -> str:
        prompt_arguments = {"label": arguments["label"]}

        for index, example in enumerate(self._examples, start=1):
            prompt_arguments[f"image_{index}"] = str(example["image_path"])
            prompt_arguments[f"label_{index}"] = example["label"]
            prompt_arguments[f"bbox_2d_{index}"] = json.dumps(
                {"bbox_2d": example["bbox_2d"]}
            )

        return self.template.render(**prompt_arguments)

    def image_paths(self) -> list[Path]:
        return self._example_image_paths

    def _extract_image_paths(self, template_content: str) -> list[Path]:
        image_paths: list[Path] = []
        image_matches: list[str] = re.findall(r"Image:\s*(.+)", template_content)
        for path in image_matches:
            image_paths.append(self._resolve_image_path(path.strip()))

        if not image_paths:
            raise ValueError(
                f"No example images found in prompt '{self.path}'. "
                "Add lines with 'Image: data/pages/with_ner/.../example.jpg'."
            )

        return image_paths

    def _resolve_image_path(self, image_path: str) -> Path:
        parsed_path = Path(image_path)
        if parsed_path.is_absolute():
            return parsed_path

        cwd_path = Path.cwd() / parsed_path
        if cwd_path.exists():
            return cwd_path

        return (self.path.parent / parsed_path).resolve()

    def _load_example_data(self, image_path: Path) -> FewShotExample:
        faces_jsonl_path = image_path.with_name(f"{image_path.stem}_faces.jsonl")
        if not faces_jsonl_path.exists():
            raise FileNotFoundError(
                f"Expected faces file '{faces_jsonl_path}' for image '{image_path}'."
            )

        lines = faces_jsonl_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"Faces file '{faces_jsonl_path}' is empty.")

        first_line = lines[0]
        face_data = json.loads(first_line)

        bounding_box = BoundingBox.from_page(
            page_width=face_data["page_width"],
            page_height=face_data["page_height"],
            page_left=face_data["page_left"],
            page_top=face_data["page_top"],
            width=face_data["width"],
            height=face_data["height"],
        )

        return {
            "image_path": image_path,
            "label": face_data["person_name"],
            "bbox_2d": bounding_box.to_list(),
        }
