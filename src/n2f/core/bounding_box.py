"""A module for representing bounding boxes."""

from dataclasses import dataclass
from typing import TypedDict


class BoundingBoxDict(TypedDict):
    """Represents a single detection dictionary."""

    bbox_2d: list[int]


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates (x_min, y_min, x_max, y_max)."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def to_dict(self) -> BoundingBoxDict:
        """Converts the bounding box to a dictionary."""
        return {"bbox_2d": [self.x_min, self.y_min, self.x_max, self.y_max]}

    def to_list(self) -> list[int]:
        """Converts the bounding box to a list of coordinates."""
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    @classmethod
    def from_page(
        cls,
        page_width: int,
        page_height: int,
        page_left: int,
        page_top: int,
        width: int,
        height: int,
    ) -> "BoundingBox":
        """
        Creates a BoundingBox instance with normalized coordinates
        from page coordinates and dimensions.
        """

        x_min = int((page_left / page_width) * 1000)
        x_max = int(((page_left + width) / page_width) * 1000)
        y_min = int((page_top / page_height) * 1000)
        y_max = int(((page_top + height) / page_height) * 1000)
        return cls(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
