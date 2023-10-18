from assistants.assistant import Assistant
from embedders.embedder import Embedder
from tools import unique
from vectordbs.vector_db import VectorDB


class KBOnlyAssistant(Assistant):
    """
    No LLM. Return naked chunks from knowledge base, ordered by relevancy.
    """

    embedder: Embedder
    vectordb: VectorDB

    def __init__(self, embedder: Embedder, vectordb: VectorDB):
        self.vectordb = vectordb
        self.embedder = embedder

    def answer_question(self, question: str, history=None):
        top_k = 6
        embedding = self.embedder.create_embedding(question)
        chunks = self.vectordb.query(embedding, top=top_k)
        texts = [c.text for c in chunks]
        answer = '\n\n'.join(texts)
        filenames = [f'* {c.filename}\n' for c in chunks]
        filenames = unique(filenames)
        sources = "\n\n## Sources:\n" + ''.join(filenames)
        answer += sources
        return answer


if __name__ == '__main__':
    from embedders.local_all_mpnet_base_v2 import LocalAllMpnetBaseV2
    from vectordbs.chroma_vector_db import ChromaVectorDB

    embedder = LocalAllMpnetBaseV2()
    vectordb = ChromaVectorDB()
    kbonly_assistant = KBOnlyAssistant(embedder=embedder, vectordb=vectordb)

    #q = 'How do you implement end-to-end traceability for data and models in SageMaker?'
    #q = 'How do you analyze models in SageMaker?'
    #q = 'How to check if an endpoint is KMS encrypted?'
    #q = 'project templates'
    #q = 'ensuring legal compliance'
    q = 'In what regions is SageMaker Edge Manager available?'
    #q = 'Buy a model'
    answer = kbonly_assistant.answer_question(q)
    print(answer)
