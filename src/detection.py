from typing import Tuple, List
from dataclasses import dataclass

from loguru import logger
import cv2
import imutils
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms


normalizer = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


@dataclass
class Parameters:
    scale: float
    step_size: int
    win_width: int
    win_height: int
    min_size_factor: int

    @property
    def dict(self):
        return {'scale': self.scale,
                'step_size': self.step_size,
                'win_width': self.win_width,
                'win_height': self.win_height,
                'min_size_factor': self.min_size_factor}


@dataclass
class Window:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def items(self):
        return self.left, self.top, self.right, self.bottom

    def as_dict(self):
        return {'left': self.left, 'top': self.top, 'right': self.right, 'bottom': self.bottom}

    def __iter__(self):
        for value in (self.left, self.top, self.right, self.bottom):
            yield value


def pyramid(image: np.ndarray, scale: float, min_size: Tuple[int, int]):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        # yield the next image in the pyramid
        yield image


def sliding_window(image: np.ndarray, step_size: int, window_size:  Tuple[int, int]):
    # slide a window across the image
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            # yield the current window
            yield x, y, image[y:y + window_size[1], x:x + window_size[0]]


def generate_windows(image: np.ndarray, scale: float, step_size: int, win_width: int, win_height: int, min_size_factor: float) -> List[Window]:
    windows = []
    # loop over the image pyramid
    i = 0
    for resized in pyramid(image, scale=scale, min_size=(int(win_width * min_size_factor), int(win_height * min_size_factor))):
        i += 1
        # loop over the sliding window for each layer of the pyramid
        for x, y, window in sliding_window(resized, step_size=step_size, window_size=(win_width, win_height)):
            if window.shape[0] < win_height or window.shape[1] < win_width:
                continue
            window = Window(*map(lambda x: int(x * scale ** i), [x, y, x + win_width, y + win_height]))
            windows.append(window)
    return windows


def create_crops_and_bbox_coordinates(image: Image.Image, windows: List[Window]) -> Tuple[np.ndarray, List[Window]]:
    crops_for_embedding = []
    coords_for_bounding_boxes = []
    for window in windows:
        window_w = window.right - window.left
        window_h = window.bottom - window.top
        if window.right > image.size[0] + window_w // 2 or window.bottom > image.size[1] + window_h // 2:
            continue
        crops_for_embedding.append(np.array(image.crop(window.items).resize((229, 229))))
        coords_for_bounding_boxes.append(window)

    crops_for_embedding = np.array(crops_for_embedding).transpose(0, 3, 1, 2)
    return crops_for_embedding, coords_for_bounding_boxes


def create_windows_embeddings(model: nn.Module, crops_for_embedding: np.ndarray, batch_size: int) -> np.ndarray:
    embeddings = []

    for shift in range(0, crops_for_embedding.shape[0], batch_size):
        batch = crops_for_embedding[shift:min(shift + batch_size, crops_for_embedding.shape[0])]

        batch = torch.Tensor(batch) / 255.
        batch = normalizer(batch)

        embeddings.append(model(batch.to(model.device)).detach().cpu().numpy())

    embeddings = np.vstack(embeddings)
    return embeddings


def create_image_windows_embeddings(image_filepath: str,
                                    model: nn.Module,
                                    batch_size: int,
                                    parameters: Parameters) -> Tuple[List[Window], np.ndarray]:

    image = cv2.imread(image_filepath)
    windows = generate_windows(image, **parameters.dict)
    logger.debug(f'Generated initial windows: {len(windows)}')

    image = Image.open(image_filepath).convert('RGB')
    crops_for_embedding, coords_for_bounding_boxes = create_crops_and_bbox_coordinates(image, windows)
    logger.debug(f'Generated crops: {len(crops_for_embedding)}')

    windows_embeddings = create_windows_embeddings(model, crops_for_embedding, batch_size)
    return coords_for_bounding_boxes, windows_embeddings
