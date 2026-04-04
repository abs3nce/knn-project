from dataclasses import dataclass

Label = str


@dataclass
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass
class AnnotatedBoundingBox:
    bounding_box: BoundingBox
    label: Label
