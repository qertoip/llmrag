import json

import boto3

from llms.llm import Llm


class AwsLlama2(Llm):

    def __init__(self):
        self.aws_client = boto3.client("sagemaker-runtime")

    def prompt(self, query: str, rag_context: str = None) -> str:
        if not rag_context:
            full_prompt = query
        else:
            full_prompt = RAG_TEMPLATE.format(context=rag_context, question=query)
        # TODO: limit rag_context to LLM context window which is 4K tokens for LLAMA-2 (minus system and template)
        dialog = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_prompt}
        ]
        payload = {
            "inputs": [dialog],
            "parameters": {"max_new_tokens": 1024, "temperature": 0.01}
        }
        output = self.aws_client.invoke_endpoint(
            EndpointName="jumpstart-dft-meta-textgeneration-llama-2-7b-f",
            ContentType="application/json",
            Body=json.dumps(payload),
            CustomAttributes="accept_eula=true",
        )
        output = output["Body"].read().decode("utf8")
        output = json.loads(output)
        answer = output[0]['generation']['content']
        return answer


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
