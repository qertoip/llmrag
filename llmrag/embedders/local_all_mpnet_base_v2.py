import os

import numpy as np
from chromadb.utils import embedding_functions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF logging (must be run before importing tf/keras/spacy)
from embedders.embedder import Embedder
from tools import unit_vector, mean_vector


class LocalAllMpnetBaseV2(Embedder):

    encoder_model = None

    def __init__(self):
        self.encoder_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-mpnet-base-v2')

    def create_embedding(self, text: str) -> np.ndarray:
        clean_text = self.clean(text)
        embedding = self.encoder_model([clean_text])[0]
        #unit_embedding = unit_vector(embedding)
        #return unit_embedding
        return embedding
