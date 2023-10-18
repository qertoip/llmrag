import numpy as np

from embedders.aws_universal_sentence_encoder_cmlm import AwsUniversalSentenceEncoderCMLM
from test import assert_is_unit_vector


def test_create_embedding():
    embedder = AwsUniversalSentenceEncoderCMLM()
    vector = embedder.create_embedding('Hello World!')
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (1024,)
    assert_is_unit_vector(vector)
