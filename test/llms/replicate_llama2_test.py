from llms.replicate_llama2 import ReplicateLlama2


def test_prompt():
    llm = ReplicateLlama2()
    answer = llm.prompt('What is 17 + 59?')
    assert '76' in answer
