import json

import boto3
import numpy as np

from tools import unit_vector
from embedders.embedder import Embedder


class AwsUniversalSentenceEncoderCMLM(Embedder):
    """
    Get embeddings by calling our own AWS SageMaker Endpoint with deployed embeddings model: Universal Sentence Encoder CMLM.

    This is just to show AWS SageMaker Endpoint integration.

    This AWS embeddings model was picked to cut down on $$$ for PoC.
    The target model might be something much more powerful like GPT-J 6B Embedding.
    """
    aws_client = None

    def __init__(self):
        self.aws_client = boto3.client('runtime.sagemaker')

    def create_embedding(self, text: str) -> np.ndarray:
        text = self.clean(text)
        response = self._query_endpoint(text)
        embedding_as_list, _ = self._parse_response(response)
        embedding = np.array(embedding_as_list)
        unit_embedding = unit_vector(embedding)
        return unit_embedding

    def _query_endpoint(self, text: str):
        encoded_text = text.encode('utf-8')
        response = self.aws_client.invoke_endpoint(
            EndpointName='jumpstart-dft-sentence-encoder-cmlm-en-large-1',
            ContentType='application/x-text',
            Body=encoded_text,
            Accept='application/json;verbose'
        )
        return response

    def _parse_response(self, response):
        model_predictions = json.loads(response['Body'].read())
        embedding, model_output = model_predictions['embedding'], model_predictions['model_output']
        return embedding, model_output
