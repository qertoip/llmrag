import json


def parse_response_falcon(query_response):
    res = json.loads(query_response)
    print(res)
    return res["generated_text"]


def parse_response_model_flan_t5(query_response):
    model_predictions = json.loads(query_response["Body"].read())
    generated_text = model_predictions["generated_texts"]
    return generated_text


def parse_response_multiple_texts_bloomz(query_response):
    generated_text = []
    model_predictions = json.loads(query_response["Body"].read())
    for x in model_predictions[0]:
        generated_text.append(x["generated_text"])
    return generated_text


MODEL_CONFIG = {
    "huggingface-llm-falcon-7b-instruct-bf16": {
        "instance type": "ml.g4dn.xlarge",
        "env": {},
        #"env": {"SAGEMAKER_MODEL_SERVER_WORKERS": "1", "TS_DEFAULT_WORKERS_PER_MODEL": "1"},
        "parse_function": parse_response_falcon,
        "prompt": """Answer based on context:\n\n{context}\n\n{question}""",
    },
    # "huggingface-text2text-flan-t5-xxl": {
    #     # "instance type": "ml.g5.12xlarge",
    #     "instance type": "ml.g4dn.2xlarge",
    #     "env": {"SAGEMAKER_MODEL_SERVER_WORKERS": "1", "TS_DEFAULT_WORKERS_PER_MODEL": "1"},
    #     "parse_function": parse_response_model_flan_t5,
    #     "prompt": """Answer based on context:\n\n{context}\n\n{question}""",
    # },
    # "huggingface-textgeneration1-bloomz-7b1-fp16": {
    #     "instance type": "ml.g5.12xlarge",
    #     "env": {},
    #     "parse_function": parse_response_multiple_texts_bloomz,
    #     "prompt": """question: \"{question}"\\n\nContext: \"{context}"\\n\nAnswer:""",
    # },
    # "huggingface-text2text-flan-ul2-bf16": {
    #     "instance type": "ml.g5.24xlarge",
    #     "env": {
    #         "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
    #         "TS_DEFAULT_WORKERS_PER_MODEL": "1"
    #     },
    #     "parse_function": parse_response_model_flan_t5,
    #     "prompt": """Answer based on context:\n\n{context}\n\n{question}""",
    # }
}
