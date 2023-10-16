import re
from abc import ABC, abstractmethod

import numpy as np

from tools import kb_path


class Embedder(ABC):

    link_in_parentheses_regex = re.compile(r'\(https://[^\s\)]+\)')

    @abstractmethod
    def create_embedding(self, text: str) -> np.ndarray:
        ...

    def clean(self, text):
        # Remove Markdown links like (https://...)
        text = re.sub(self.link_in_parentheses_regex, '', text)
        for stopword in ['**Note**', '**Topics**', '**', '#### ', '### ', '## ', '[', ']']:
            text = text.replace(stopword, '')
        return text


# class Concrete(Embedder):
#     def create_embedding(self, text: str) -> np.ndarray:
#         pass
#
#
# if __name__ == '__main__':
#     e = Concrete()
#     text = (kb_path() / 'aws-properties-sagemaker-inferenceexperiment-capturecontenttypeheader.md').read_text()
#     text = e.clean(text)
#     print(text)
