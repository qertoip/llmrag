from abc import ABC, abstractmethod


class Assistant(ABC):

    @abstractmethod
    def answer_question(self, question: str, history=None) -> str:
        ...

    def __call__(self, *args, **kwargs):
        return self.answer_question(*args)
