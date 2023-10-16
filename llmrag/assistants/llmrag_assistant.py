import replicate

from assistants.assistant import Assistant
from embedders.embedder import Embedder
from tools import root_path
from vectordbs.vector_db import VectorDB


class LlmRagAssistant(Assistant):
    """
    Retrieval Augmented Generation (RAG) from internal knowledge base (KB) with LLM.

    LLAMA-2 via Replicate API to save $$$ for PoC.
    Can't be used in production as target dataset cant be shared externally.
    """

    client: replicate.client.Client

    def __init__(self, embedder: Embedder, vectordb: VectorDB):
        self.vectordb = vectordb
        self.embedder = embedder
        api_key = (root_path() / 'config' / 'replicate_api_key.txt').read_text().strip()
        self.client = replicate.client.Client(api_token=api_key)

    def answer_question(self, question: str, history=None):
        embedding = self.embedder.create_embedding(question)
        chunks = self.vectordb.query(embedding, top=10)
        texts = [c.text for c in chunks]
        filenames = [f'* {c.filename}\n' for c in chunks]

        rag_context = '\n\n'.join(texts)

        print(f'rag_context={rag_context}')

        # TODO: limit rag_context to LLM context window which is 4K tokens for LLAMA-2 (minus system and template)

        output = self.client.run(
            "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e",
            input={
                'system': SYSTEM_PROMPT,
                'prompt': RAG_TEMPLATE.format(context=rag_context, question=question),
                'max_new_tokens': 1024,
            },
           # stream=True
        )
        full_output = ''
        for item in output:
            # https://replicate.com/meta/llama-2-7b-chat/versions/8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e/api#output-schema
            # yield item
            full_output += item

        sources = "\n\n## Sources:\n" + ''.join(filenames)
        full_output += sources

        return full_output


SYSTEM_PROMPT = """
Always answer as helpfully and truthfully as possible.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, don't share false information.
Only answer about Amazon AWS. 
""".strip()

RAG_TEMPLATE = """
Use the following pieces of context to answer the question at the end.
Context: {context}
Question pertaining AWS SageMaker: {question}
""".strip()
