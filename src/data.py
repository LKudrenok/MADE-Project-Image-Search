from __future__ import annotations
import os
import json
from typing import List
from pathlib import Path
from dataclasses import dataclass

from loguru import logger
import numpy as np
from torch import nn
from torchvision import transforms
from PIL import Image

from .detection import Window, create_image_windows_embeddings
from .search import Neighbour


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(229, 229)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


@dataclass
class DatabaseItem:
    path: str
    windows: List[Window]
    embeddings: np.ndarray
    start_index: int
    end_index: int


class Database:
    FILENAME_JSON = 'database.json'
    FILENAME_NPY = 'database.npy'

    def __init__(self):
        self.items: List[DatabaseItem] = []

    def __getitem__(self, index: int) -> DatabaseItem:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    @classmethod
    def from_folder(cls, folder: str) -> Database:
        database = Database()
        database.load(folder)
        return database

    @property
    def embeddings(self):
        return np.vstack([item.embeddings for item in self.items])

    def append(self, item: DatabaseItem):
        self.items.append(item)

    def save(self, folder: str):
        Path(folder).mkdir(exist_ok=True)
        with open(Path(folder, self.FILENAME_JSON), 'w') as file:
            json.dump(
                fp=file,
                obj={
                    'paths': [item.path for item in self.items],
                    'windows': [[win.items for win in item.windows] for item in self.items],
                    'start_index': [item.start_index for item in self.items],
                    'end_index': [item.end_index for item in self.items]
                },
                ensure_ascii=False,
                indent=4
            )
        np.save(str(Path(folder, self.FILENAME_NPY)), [item.embeddings for item in self.items])
        logger.info(f'Database saved into {folder}')

    def load(self, folder: str):
        with open(Path(folder, self.FILENAME_JSON), 'r') as file:
            loaded_json = json.load(file)
        loaded_npy = np.load(str(Path(folder, self.FILENAME_NPY)), allow_pickle=True)
        self.items = []
        for i, embedding in enumerate(loaded_npy):
            item = DatabaseItem(
                path=loaded_json['paths'][i],
                windows=[Window(*coords) for coords in loaded_json['windows'][i]],
                embeddings=embedding,
                start_index=loaded_json['start_index'][i],
                end_index=loaded_json['end_index'][i]
            )
            self.items.append(item)
        logger.info(f'Database loaded from {folder}, items number - {len(self.items)}')


def open_image(filepath: str) -> Image.Image:
    with open(filepath, 'rb') as f:
        image = Image.open(f).convert('RGB')
    return image


def image_to_encoding(model: nn.Module, filepath: str = None, image: Image.Image = None):
    if filepath:
        image = open_image(filepath)
    image = transform(image)
    image = image.unsqueeze(0)
    embeddings = model(image.to(model.device)).detach().cpu().numpy()
    return embeddings[0]


def fill_neighbours(neighbours: List[Neighbour], database: Database) -> List[Neighbour]:
    for i, n in enumerate(neighbours):
        for item in database:
            if item.start_index <= n.index <= item.end_index:
                n.path = item.path
                n.window = item.windows[n.index - item.start_index]
                neighbours[i] = n
    return neighbours


def create_database(root: str, model: nn.Module, progress_bar: bool = False, **detection_params) -> Database:
    st_progress_bar = None
    if progress_bar:
        import streamlit as st
        st_progress_bar = st.progress(0.0)

    database = Database()

    start_index = 0
    filenames = os.listdir(root)
    for i, image_filepath in enumerate(filenames, 1):
        image_filepath = str(Path(root, image_filepath))
        logger.info(f'Processing {image_filepath} ({i}/{len(filenames)})')

        coords_for_bounding_boxes, windows_embeddings = create_image_windows_embeddings(image_filepath, model, **detection_params)
        logger.info(f'Number of windows/embeddings: {len(coords_for_bounding_boxes)}')

        end_index = start_index + len(coords_for_bounding_boxes) - 1
        item = DatabaseItem(image_filepath, coords_for_bounding_boxes, windows_embeddings, start_index, end_index)
        database.append(item)
        start_index = end_index + 1

        if progress_bar:
            st_progress_bar.progress(i / len(filenames))

    logger.info(f'Database created with {len(database)} items')
    return database
