from __future__ import annotations
from typing import List

from loguru import logger

from .search import Neighbour


EPS = 1e-6


class BoundingBox:
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.width = abs(self.x1 - self.x2)
        self.height = abs(self.y1 - self.y2)

    @property
    def area(self):
        return (self.width + EPS) * (self.height + EPS)

    def contains(self, other: BoundingBox):
        return self.x1 < other.x1 < other.x2 < self.x2 and self.y1 < other.y1 < other.y2 < self.y2

    def intersect(self, other: BoundingBox) -> float:
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        intersection = max(0.0, x2 - x1 + EPS) * max(0.0, y2 - y1 + EPS)
        return intersection

    def iou(self, other: BoundingBox) -> float:
        intersection = self.intersect(other)
        iou = intersection / float(self.area + other.area - intersection)
        return iou


def filter_neighbours(neighbours: List[Neighbour], threshold_iou: float) -> List[Neighbour]:
    filtered = []
    proposals = neighbours.copy()
    removed = set()
    for p in proposals:
        if p in removed:
            continue
        current_bbox = BoundingBox(*p.window)
        for other in proposals:
            if other == p or other in removed:
                continue
            other_bbox = BoundingBox(*other.window)
            iou = current_bbox.iou(other_bbox)
            # logger.debug(f'{p.window} vs {other.window}: iou = {iou}; best contains worst = {current_bbox.contains(other_bbox)}')
            if iou > threshold_iou or current_bbox.contains(other_bbox) or other_bbox.contains(current_bbox):
                removed.add(other)
        filtered.append(p)
    logger.info(f'Keep {len(filtered)} windows from {len(neighbours)}')
    return filtered
