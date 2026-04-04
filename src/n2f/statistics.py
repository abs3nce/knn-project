from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict
import json

from n2f.annotation_result import AnnotationResult, AnnotationResultDict


class StatisticsDict(TypedDict):
    image_path: str
    image_id: str
    model: str
    prompt_path: str
    raw_response: str
    annotation_result: AnnotationResultDict
    success: int
    error_message: str
    start_timestamp: str
    end_timestamp: str
    total_time_seconds: float
    tokens_used: int


@dataclass
class Statistics:
    image_path: Path
    image_id: str
    model: str
    prompt_path: Path
    raw_response: str
    annotation_result: AnnotationResult
    success: bool
    error_message: str | None
    start_timestamp: datetime
    end_timestamp: datetime
    total_time: timedelta
    tokens_used: int

    def to_dict(self) -> StatisticsDict:
        return {
            "image_path": str(self.image_path),
            "image_id": self.image_id,
            "model": self.model,
            "prompt_path": str(self.prompt_path),
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
        return json.dumps(self.to_dict())
