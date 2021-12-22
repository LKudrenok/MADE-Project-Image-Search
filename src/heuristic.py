import numpy as np
import cv2
import albumentations as a
from PIL import Image
from loguru import logger

from .data import image_to_encoding
from .networks import Model
from .search import Index


def apply_augmentation(image, augmenter):
    return augmenter(image=np.array(image))['image']


def calculate_heuristic(model: Model, query_image: Image.Image, n: int = 100) -> float:
    logger.debug(f'Creating {n} augmented images...')

    augmentations = [
        a.HorizontalFlip(p=0.5),
        a.RandomBrightnessContrast(always_apply=True),
        a.ShiftScaleRotate(rotate_limit=359, scale_limit=(-0.4, 0.4), shift_limit=0.05, border_mode=cv2.BORDER_REPLICATE, always_apply=True)
    ]

    augmented_images = []
    for _ in range(n):
        temp_image = query_image.copy()
        for augmenter in augmentations:
            temp_image = apply_augmentation(temp_image, augmenter)
        augmented_images.append(temp_image)

    embeddings = np.vstack([image_to_encoding(model, image=x) for x in augmented_images])

    image_embedding = image_to_encoding(model, image=query_image)

    index = Index()
    index.build(embeddings, verbose=False)
    neighbours = index.search(image_embedding)

    _scores = np.array([n.score for n in neighbours])
    _mean = _scores.mean()
    _std = _scores.std()
    _max = _scores.max()
    logger.debug(f'Neighbours statistics: mean: {_mean:.4f}, std: {_std:.4f}, max: {_max:.4f}, mean + 3 * std: {_mean + 3 * _std:.4f}')

    return min(_mean + 3 * _std, _max)
