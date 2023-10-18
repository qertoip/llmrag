import hashlib
import sys
from pathlib import Path
import logging
from typing import Iterable

import numpy as np
from numpy.linalg import norm

BOLD, UNBOLD = '\033[1m', '\033[0m'


def root_path() -> Path:
    return Path(__file__).parent.parent.absolute()


def kb_path() -> Path:
    path = root_path() / 'db' / 'kb'
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_path() -> Path:
    path = root_path() / 'log'
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(filename: str):
    file_handler = logging.FileHandler(filename=log_path() / filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stdout_handler],
        encoding='utf-8'
    )


def unit_vector(vector: np.ndarray):
    l2_norm = np.linalg.norm(vector, ord=2)
    if l2_norm > 0:
        return vector / l2_norm
    else:
        return vector


def mean_vector(vectors: list):
    np_vectors = np.array(vectors)
    return np.mean(np_vectors, axis=0)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot_product = v1.dot(v2)
    length_normalization = norm(v1, ord=2) * norm(v2, ord=2)  # L2
    return dot_product / length_normalization


def chunk_id(s: str):
    return hashlib.sha3_256(s.encode('utf-8')).hexdigest()


def unique(iterable: Iterable) -> list:
    """Return unique elements while maintaining order (can't simply use set)."""
    return list(dict.fromkeys(iterable if iterable is not None else []))
