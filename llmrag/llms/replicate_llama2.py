import replicate

from llms.llm import Llm
from tools import root_path


class ReplicateLlama2(Llm):

    def __init__(self):
        api_key = (root_path() / 'config' / 'replicate_api_key.txt').read_text().strip()
        self.client = replicate.client.Client(api_token=api_key)

    def prompt(self, query: str, rag_context: str = None) -> str:
        if not rag_context:
            full_prompt = query
        else:
            full_prompt = RAG_TEMPLATE.format(context=rag_context, question=query)
        # TODO: limit rag_context to LLM context window which is 4K tokens for LLAMA-2 (minus system and template)
        output = self.client.run(
            "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e",
            input={
                'system': SYSTEM_PROMPT,
                'prompt': full_prompt,
                'max_new_tokens': 1024,
                'temperature': 0.01,  # make deterministic (exact 0 not accepted)
            },
            # stream=True
        )
        full_answer = ''
        for item in output:
            # https://replicate.com/meta/llama-2-7b-chat/versions/8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e/api#output-schema
            # yield item
            full_answer += item
        return full_answer


SYSTEM_PROMPT = """
Always answer helpfully and truthfully.
If a question does not make any sense or is incoherent, explain why.
If you don't know the answer to a question, don't share false information.
Skip the nice intro about being helpful or happy. 
""".strip()

RAG_TEMPLATE = """
Use the following context to answer the question at the end.
Context:

{context}

Question on AWS SageMaker:

{question}
""".strip()
