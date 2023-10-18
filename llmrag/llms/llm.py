from abc import ABC, abstractmethod


class Llm(ABC):

    @abstractmethod
    def prompt(self, query: str, rag_context: str) -> str:
        ...
