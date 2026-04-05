"""A module for representing bounding boxes and annotated bounding boxes."""

from dataclasses import dataclass

Label = str


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates (x_min, y_min, x_max, y_max)."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass
class AnnotatedBoundingBox:
    """Represents a bounding box with a label."""

    bounding_box: BoundingBox
    label: Label
