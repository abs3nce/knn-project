from dataclasses import dataclass
import json
from typing import TypedDict

from n2f.bounding_box import AnnotatedBoundingBox, BoundingBox


class DetectionDict(TypedDict):
    bounding_box: list[int]
    label: str


class AnnotationResultDict(TypedDict):
    detections: list[DetectionDict]


@dataclass
class AnnotationResult:
    annotated_bounding_boxes: list[AnnotatedBoundingBox]

    @classmethod
    def from_dict(cls, data: AnnotationResultDict) -> "AnnotationResult":
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
            annotated_bounding_boxes.append(
                AnnotatedBoundingBox(
                    bounding_box=bounding_box,
                    label=label,
                )
            )

        return cls(annotated_bounding_boxes=annotated_bounding_boxes)

    @classmethod
    def from_json(cls, json_string: str) -> "AnnotationResult":
        data: AnnotationResultDict = json.loads(json_string)
        print(data)
        return cls.from_dict(data)
