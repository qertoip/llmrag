import replicate

from assistants.assistant import Assistant
from embedders.embedder import Embedder
from llms.llm import Llm
from tools import root_path, unique
from vectordbs.vector_db import VectorDB


class LlmRagAssistant(Assistant):
    """
    Retrieval Augmented Generation (RAG) from internal knowledge base (KB) with LLM.

    LLAMA-2 via Replicate API to save $$$ for PoC.
    Can't be used in production as target dataset cant be shared externally.
    """

    def __init__(self, embedder: Embedder, vectordb: VectorDB, llm: Llm):
        self.vectordb = vectordb
        self.embedder = embedder
        self.llm = llm

    def answer_question(self, question: str, history=None):
        top_k = 10
        embedding = self.embedder.create_embedding(question)
        chunks = self.vectordb.query(embedding, top=top_k)
        texts = [c.text for c in chunks]
        filenames = [f'* {c.filename}\n' for c in chunks]
        filenames = unique(filenames)[0:4]  # only present the top sources to make it brief?
        rag_context = '\n\n'.join(texts)
        answer = self.llm.prompt(query=question, rag_context=rag_context)
        sources = "\n\n#### Sources:\n" + ''.join(filenames)
        answer += sources
        return answer
