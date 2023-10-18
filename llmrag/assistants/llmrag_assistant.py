import replicate

from assistants.assistant import Assistant
from embedders.embedder import Embedder
from tools import root_path, unique
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
        top_k = 10
        embedding = self.embedder.create_embedding(question)
        chunks = self.vectordb.query(embedding, top=top_k)
        texts = [c.text for c in chunks]
        filenames = [f'* {c.filename}\n' for c in chunks]
        filenames = unique(filenames)[0:4]  # only present the top sources to make it brief?
        rag_context = '\n\n'.join(texts)
        print(f'rag_context={rag_context}')

        # TODO: limit rag_context to LLM context window which is 4K tokens for LLAMA-2 (minus system and template)

        output = self.client.run(
            "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e",
            input={
                'system': SYSTEM_PROMPT,
                'prompt': RAG_TEMPLATE.format(context=rag_context, question=question),
                'max_new_tokens': 1024,
                'temperature': 0.01,  # make deterministic (exact 0 not accepted)
            },
           # stream=True
        )
        full_output = ''
        for item in output:
            # https://replicate.com/meta/llama-2-7b-chat/versions/8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e/api#output-schema
            # yield item
            full_output += item

        sources = "\n\n#### Sources:\n" + ''.join(filenames)
        full_output += sources

        return full_output


SYSTEM_PROMPT = """
Always answer helpfully and truthfully.
If a question does not make any sense or is incoherent, explain why.
If you don't know the answer to a question, don't share false information.
Skip the nice intro about being helpful or happy. 
""".strip()

RAG_TEMPLATE = """
Use the following context as much as possible to answer the question at the end.
Context:

{context}

Question on AWS SageMaker:

{question}
""".strip()
