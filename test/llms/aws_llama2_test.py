from llms.aws_llama2 import AwsLlama2


def test_prompt():
    llm = AwsLlama2()
    answer = llm.prompt('What is 17 + 59?')
    assert '76' in answer
