import logging as log
import os

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF logging (must be run before importing tf/keras/spacy)
import spacy_universal_sentence_encoder

from tools import unit_vector, mean_vector
from embedders.embedder import Embedder


class LocalUniversalSentenceEncoder(Embedder):

    encoder_model = None

    def __init__(self):
        # load one of the models: ['en_use_md', 'en_use_lg', 'xx_use_md', 'xx_use_lg']
        #variant = 'en_use_lg'
        variant = 'en_use_cmlm_lg'
        log.info(f'Loading model from tfhub.dev ({variant})')
        self.encoder_model = spacy_universal_sentence_encoder.load_model(variant)

    def create_embedding(self, text: str) -> np.ndarray:
        # sentences = text.split('.')
        # embeddings = []
        # for sentence in sentences:
        #     sentence = sentence.strip()
        #     for stopword in ['**Note**', '**Topics**', '**']:
        #         sentence = sentence.replace(stopword, '')
        #     if len(sentence) < 10:
        #         continue
        #     #print(f'>>>>{sentence}<<<<')
        #     doc = self.encoder_model(text)
        #     embedding = doc.vector
        #     embeddings.append(embedding)
        # mean_embedding = mean_vector(embeddings)
        # unit_embedding = unit_vector(mean_embedding)

        for stopword in ['**Note**', '**Topics**', '**']:
            text = text.replace(stopword, '')
        embedding = self.encoder_model(text)
        unit_embedding = unit_vector(embedding)
        return unit_embedding
