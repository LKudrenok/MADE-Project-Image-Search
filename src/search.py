from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger
import faiss
import numpy as np

from .detection import Window


@dataclass
class Neighbour:
    index: int
    score: float
    path: Optional[str] = None
    window: Optional[Window] = None

    def __hash__(self):
        return hash(self.index)


class Index:
    FILENAME = 'index.bin'

    def __init__(self):
        self.index = faiss.IndexFlatL2(0)

    def __str__(self):
        return f'faiss.IndexFlatL2 (size: {self.n_items}x{self.embedding_size})'

    def __repr__(self):
        return str(self)

    @property
    def n_items(self) -> int:
        return self.index.ntotal

    @property
    def embedding_size(self) -> int:
        return self.index.d

    def build(self, embeddings: np.ndarray, verbose: bool = True):
        size = embeddings.shape[-1]
        self.index = faiss.IndexFlatL2(size)
        self.index.add(embeddings)
        if verbose:
            logger.info(f'Index built (size: {self.n_items} x {self.embedding_size})')

    def save(self, folder: str):
        Path(folder).mkdir(exist_ok=True)
        filepath = str(Path(folder, self.FILENAME))
        faiss.write_index(self.index, filepath)
        logger.info(f'Index saved into {folder}')

    def load(self, folder: str):
        filepath = str(Path(folder, self.FILENAME))
        self.index = faiss.read_index(filepath)
        logger.info(f'Index loaded from {folder} (size: {self.n_items} x {self.embedding_size})')

    @classmethod
    def from_folder(cls, folder: str) -> Index:
        index = Index()
        index.load(folder)
        return index

    def search(self, query: np.ndarray, n_neighbors: int = None) -> List[Neighbour]:
        query = query.reshape(1, query.size)
        n_neighbors = n_neighbors or self.n_items
        scores, indices = self.index.search(x=query, k=n_neighbors)
        neighbours = [Neighbour(i, s) for i, s in zip(indices[0], scores[0])]
        return neighbours

