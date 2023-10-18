import numpy as np
from pytest import approx


def assert_is_unit_vector(vector):
    assert np.linalg.norm(vector, ord=2) == approx(1)
