from replicate.client import Client

from tools import root_path


SYSTEM_PROMPT = """
Always answer as helpfully and truthfully as possible.
Skip the niceties and go straight to the point.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.
"""


def answer_question(question: str, history):
    api_key = (root_path() / 'config' / 'replicate_api_key.txt').read_text().strip()
    client = Client(api_token=api_key)
    output = client.run(
        "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e",
        input={
            'system': SYSTEM_PROMPT,
            'prompt': question,
            'max_new_tokens': 512,
        },
       # stream=True
    )
    full_output = ''
    for item in output:
        # https://replicate.com/meta/llama-2-7b-chat/versions/8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e/api#output-schema
        # yield item
        full_output += item

    return full_output
