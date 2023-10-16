from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Item:
    text: str
    filename: str
    distance: float
    pass


class VectorDB(ABC):

    @abstractmethod
    def insert(self, id: str, embedding: list | np.ndarray, text: str, metadata: dict):
        ...

    @abstractmethod
    def query(self, query_embedding: list | np.ndarray, top: int = 1) -> list[Item]:
        ...

    @abstractmethod
    def __contains__(self, chunk: str) -> bool:
        ...

    @abstractmethod
    def report(self):
        ...
