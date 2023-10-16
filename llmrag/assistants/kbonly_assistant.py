from assistants.assistant import Assistant
from embedders.embedder import Embedder
from embedders.local_universal_sentence_encoder import LocalUniversalSentenceEncoder
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
        embedding = self.embedder.create_embedding(question)
        chunks = self.vectordb.query(embedding, top=20)
        texts = [c.text for c in chunks]
        filenames = [c.filename for c in chunks]
        answer = '\n\n'.join(texts)
        answer = answer[0 : 4096-256]
        #answer += "\n\nSources: " + ', '.join(filenames)
        return answer


if __name__ == '__main__':
    from embedders.aws_universal_sentence_encoder_cmlm import AwsUniversalSentenceEncoderCMLM
    from vectordbs.chroma_vector_db import ChromaVectorDB
    embedder = AwsUniversalSentenceEncoderCMLM()
    #embedder = LocalUniversalSentenceEncoder()
    vectordb = ChromaVectorDB()
    kbonly_assistant = KBOnlyAssistant(embedder=embedder, vectordb=vectordb)
    #q = 'How do you implement end-to-end traceability for data and models in SageMaker?'
    #q = 'How do you monitor and analyze metrics for models that have been deployed through SageMaker?'
    #q = 'Guidelines for MLOps'
    q = 'How to ensure compliance validation'
    answer = kbonly_assistant.answer_question(q)
    print(answer)
