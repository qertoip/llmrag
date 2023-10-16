import json

import boto3
import numpy as np

from tools import unit_vector
from embedders.embedder import Embedder


class AwsUniversalSentenceEncoderCMLM(Embedder):
    
    aws_client = None

    def __init__(self):
        self.aws_client = boto3.client('runtime.sagemaker')

    def create_embedding(self, text: str) -> np.ndarray:
        for stopword in ['**Note**', '**Topics**', '**']:
            text = text.replace(stopword, '')
        response = self._query_endpoint(text)
        embedding, _ = self._parse_response(response)
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
