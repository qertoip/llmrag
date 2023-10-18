import os

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF logging (must be run before importing tf/keras)
from chromadb.utils import embedding_functions
from embedders.embedder import Embedder


class LocalAllMpnetBaseV2(Embedder):
    """
    Get embeddings from local embeddings model all-mpnet-base-v2.
    Great for quick local experimentation w/o incurring costs.
    """

    encoder_model = None

    def __init__(self):
        self.encoder_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-mpnet-base-v2')

    def create_embedding(self, text: str) -> np.ndarray:
        clean_text = self.clean(text)
        # all-mpnet-base-v2 returns unit vectors so no need to normalize
        embedding_seq = self.encoder_model([clean_text])[0]
        embedding = np.array(embedding_seq)
        return embedding
