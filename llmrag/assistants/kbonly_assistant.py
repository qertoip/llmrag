from assistants.assistant import Assistant
from embedders.embedder import Embedder
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
        chunks = self.vectordb.query(embedding, top=6)
        texts = [c.text for c in chunks]
        answer = '\n\n'.join(texts)
        filenames = [f'* {c.filename}\n' for c in chunks]
        sources = "\n\n## Sources:\n" + ''.join(filenames)
        answer += sources
        return answer

        #answer = answer[0 : 4096-256]


if __name__ == '__main__':
    from embedders.aws_universal_sentence_encoder_cmlm import AwsUniversalSentenceEncoderCMLM
    from vectordbs.chroma_vector_db import ChromaVectorDB
    embedder = AwsUniversalSentenceEncoderCMLM()
    #embedder = LocalUniversalSentenceEncoder()
    vectordb = ChromaVectorDB()
    kbonly_assistant = KBOnlyAssistant(embedder=embedder, vectordb=vectordb)
    #q = 'How do you implement end-to-end traceability for data and models in SageMaker?'
    #q = 'How do you analyze models in SageMaker?'
    #q = 'How to check if an endpoint is KMS encrypted?'
    #q = 'project templates'
    #q = 'ensuring legal compliance'
    #q = 'What are all AWS regions where SageMaker is available?'
    #q = 'Buy a model'
    answer = kbonly_assistant.answer_question(q)
    print(answer)


# template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
# {context}
# Question: {question}
# Helpful Answer:"""
