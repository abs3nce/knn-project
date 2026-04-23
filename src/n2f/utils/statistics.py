"""A module for defining the Statistics class and related types."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict
import json

from n2f.core.bounding_box import BoundingBox
from n2f.core.annotation_result import AnnotationResult


class StatisticsDict(TypedDict):
    """Representation of the Statistics class for JSON serialization."""

    image_path: str
    image_id: str
    model: str
    prompt_path: str
    label: str
    expected_bounding_box: list[int]
    raw_response: str
    annotation_result: list[int]
    success: int
    error_message: str
    start_timestamp: str
    end_timestamp: str
    total_time_seconds: float
    tokens_used: int


@dataclass
class Statistics:
    """A class for storing statistics about the annotation process."""

    image_path: Path
    image_id: str
    model: str
    prompt_path: Path
    label: str
    expected_bounding_box: BoundingBox
    raw_response: str
    annotation_result: AnnotationResult
    success: bool
    error_message: str | None
    start_timestamp: datetime
    end_timestamp: datetime
    total_time: timedelta
    tokens_used: int

    def to_dict(self) -> StatisticsDict:
        """Returns the statistics as a dictionary."""
        return {
            "image_path": str(self.image_path),
            "image_id": self.image_id,
            "model": self.model,
            "prompt_path": str(self.prompt_path),
            "label": self.label,
            "expected_bounding_box": self.expected_bounding_box.to_list(),
            "raw_response": self.raw_response,
            "annotation_result": self.annotation_result.to_dict(),
            "success": 0 if not self.success else 1,
            "error_message": self.error_message or "",
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat(),
            "total_time_seconds": self.total_time.total_seconds(),
            "tokens_used": self.tokens_used,
        }

    def to_json(self) -> str:
        """Returns the statistics as a JSON string."""
        return json.dumps(self.to_dict())
