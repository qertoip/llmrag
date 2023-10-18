import numpy as np

from embedders.local_all_mpnet_base_v2 import LocalAllMpnetBaseV2
from test import assert_is_unit_vector


def test_create_embedding():
    embedder = LocalAllMpnetBaseV2()
    vector = embedder.create_embedding('Hello World!')
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (768,)
    assert_is_unit_vector(vector)
