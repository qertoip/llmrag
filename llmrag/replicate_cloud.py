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
    # The meta/llama-2-7b-chat model can stream output as it's running.
    # The predict method returns an iterator, and you can iterate over that output.
    for item in output:
        # https://replicate.com/meta/llama-2-7b-chat/versions/8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e/api#output-schema
        print(item, end="")


if __name__ == '__main__':
    main()
