from pathlib import Path


def root_path() -> Path:
    return Path(__file__).parent.parent.absolute()


# def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
#     dot_product = v1.dot(v2)
#     length_normalization = norm(v1) * norm(v2)  # L2
#     return dot_product / length_normalization
