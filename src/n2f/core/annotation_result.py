"""A module for representing the annotation results."""

from dataclasses import dataclass
import json

from n2f.core.bounding_box import BoundingBox, BoundingBoxDict


AnnotationResultDict = BoundingBoxDict


@dataclass
class AnnotationResult:
    """Represents the annotation result containing a list of annotated bounding boxes."""

    bounding_box: BoundingBox

    @classmethod
    def from_dict(cls, data: BoundingBoxDict) -> "AnnotationResult":
        """Returns an AnnotationResult instance created from a dictionary."""
        bounding_box_data = data["bbox_2d"]

        # Assuming the bounding box values are in the format [y_min, x_min, y_max, x_max]
        bounding_box = BoundingBox(
            x_min=bounding_box_data[1],
            y_min=bounding_box_data[0],
            x_max=bounding_box_data[3],
            y_max=bounding_box_data[2],
        )

        return cls(bounding_box=bounding_box)

    @classmethod
    def from_json(cls, json_string: str) -> "AnnotationResult":
        """Returns an AnnotationResult instance created from a JSON string."""
        data: BoundingBoxDict = json.loads(json_string)
        return cls.from_dict(data)

    @classmethod
    def empty(cls) -> "AnnotationResult":
        """Returns an empty AnnotationResult instance."""
        return cls(bounding_box=BoundingBox(0, 0, 0, 0))

    def to_dict(self) -> list[int]:
        """Returns the annotation result as a dictionary."""
        return self.bounding_box.to_list()
