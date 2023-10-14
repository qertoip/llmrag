import json

from llmrag.config import MODEL_CONFIG

question = "Which instances can I use with Managed Spot Training in SageMaker?"

payload = {
    "text_inputs": question,
    "max_length": 100,
    "num_return_sequences": 1,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": True,
}


for model_id in MODEL_CONFIG:
    endpoint_name = MODEL_CONFIG[model_id]["endpoint_name"]
    query_response = query_endpoint_with_json_payload(
        json.dumps(payload).encode("utf-8"), endpoint_name=endpoint_name
    )
    generated_texts = MODEL_CONFIG[model_id]["parse_function"](query_response)
    print(f"For model: {model_id}, the generated output is: {generated_texts[0]}\n")
