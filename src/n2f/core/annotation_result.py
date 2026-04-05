"""A module for representing the annotation results."""

from dataclasses import dataclass
from typing import TypedDict
import json

from n2f.core.bounding_box import AnnotatedBoundingBox, BoundingBox


class DetectionDict(TypedDict):
    """Represents a single detection dictionary."""

    bounding_box: list[int]
    label: str


class AnnotationResultDict(TypedDict):
    """Represents the annotation result as a dictionary."""

    detections: list[DetectionDict]


@dataclass
class AnnotationResult:
    """Represents the annotation result containing a list of annotated bounding boxes."""

    annotated_bounding_boxes: list[AnnotatedBoundingBox]

    @classmethod
    def from_dict(cls, data: AnnotationResultDict) -> "AnnotationResult":
        """Returns an AnnotationResult instance created from a dictionary."""
        annotated_bounding_boxes: list[AnnotatedBoundingBox] = []
        bounding_boxes = data["detections"]
        for bounding_box in bounding_boxes:
            bounding_box_data = bounding_box["bounding_box"]
            label = bounding_box["label"]
            bounding_box = BoundingBox(
                x_min=bounding_box_data[0],
                y_min=bounding_box_data[1],
                x_max=bounding_box_data[2],
                y_max=bounding_box_data[3],
            )

            if any(not (0 <= value <= 1000) for value in bounding_box_data):
                raise ValueError(
                    f"Bounding box values must be between 0 and 1000. "
                    f"Got {bounding_box_data}."
                )

            annotated_bounding_boxes.append(
                AnnotatedBoundingBox(
                    bounding_box=bounding_box,
                    label=label,
                )
            )

        return cls(annotated_bounding_boxes=annotated_bounding_boxes)

    @classmethod
    def from_json(cls, json_string: str) -> "AnnotationResult":
        """Returns an AnnotationResult instance created from a JSON string."""
        data: AnnotationResultDict = json.loads(json_string)
        return cls.from_dict(data)

    @classmethod
    def empty(cls) -> "AnnotationResult":
        """Returns an empty AnnotationResult instance."""
        return cls(annotated_bounding_boxes=[])

    def to_dict(self) -> AnnotationResultDict:
        """Returns the annotation result as a dictionary."""
        return {
            "detections": [
                {
                    "bounding_box": [
                        detection.bounding_box.x_min,
                        detection.bounding_box.y_min,
                        detection.bounding_box.x_max,
                        detection.bounding_box.y_max,
                    ],
                    "label": detection.label,
                }
                for detection in self.annotated_bounding_boxes
            ]
        }
