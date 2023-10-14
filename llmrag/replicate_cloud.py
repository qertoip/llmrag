from llmrag.tools import root_path

from replicate.client import Client


def main():
    api_key = (root_path() / 'config' / 'replicate_api_key.txt').read_text().strip()
    client = Client(api_token=api_key)
    output = client.run(
        "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e",
        input={
            'prompt': 'Explain what is AWS SageMaker in a form of a poem.',
            'max_new_tokens': 1024,
        }
    )
    for item in output:
        # https://replicate.com/meta/llama-2-7b-chat/versions/8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e/api#output-schema
        yield item


if __name__ == '__main__':
    main()
