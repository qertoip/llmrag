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
        # TODO: this is very basic, more can be done to clean for embedding generation

        # Remove Markdown links like (https://...)
        text = re.sub(self.link_in_parentheses_regex, '', text)

        # Remove Markdown formatting and some useless words
        for noise in ['**Note**', '**Topics**', '**', '#### ', '### ', '## ', '[', ']']:
            text = text.replace(noise, '')
        return text
