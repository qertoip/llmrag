import replicate

from assistants.assistant import Assistant
from tools import root_path


class LlmOnlyAssistant(Assistant):
    """
    Plain LLM **not** aware of any local knowledge base (KB).
    Use as a baseline for assessing KB-aware assistants.

    LLAMA-2 via Replicate API to save $$$ for PoC.
    Can't be used in production as target dataset cant be shared externally.
    """

    client: replicate.client.Client

    def __init__(self):
        api_key = (root_path() / 'config' / 'replicate_api_key.txt').read_text().strip()
        self.client = replicate.client.Client(api_token=api_key)

    def answer_question(self, question: str, history=None):
        output = self.client.run(
            "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e",
            input={
                'system': SYSTEM_PROMPT,
                'prompt': 'Pertains AWS SageMaker. ' + question,
                'max_new_tokens': 1024,
            },
           # stream=True
        )
        full_output = ''
        for item in output:
            # https://replicate.com/meta/llama-2-7b-chat/versions/8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e/api#output-schema
            # yield item
            full_output += item
        return full_output


SYSTEM_PROMPT = """
Always answer as helpfully and truthfully as possible.
Skip "happy to help". Go straight to the point.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, don't share false information.
Only answer about Amazon AWS. 
"""
