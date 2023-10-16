from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):

    @abstractmethod
    def create_embedding(self, text: str) -> np.ndarray:
        ...
