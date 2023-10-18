import sagemaker, boto3, json
from sagemaker.session import Session
from sagemaker.model import Model
from sagemaker import image_uris, model_uris, script_uris, hyperparameters
from sagemaker.predictor import Predictor
from sagemaker.utils import name_from_base


def main():
    sagemaker_session = Session()

    aws_role = sagemaker_session.get_caller_identity_arn()
    aws_role = 'arn:aws:iam::560991781189:role/AllowSageMakerDoThings'
    print(f'aws_role={aws_role}')

    aws_region = boto3.Session().region_name
    print(f'aws_region={aws_region}')

    model_version = "*"

    for model_id in MODEL_CONFIG:
        print(f'model_id={model_id}')

        endpoint_name = name_from_base(f"jumpstart-example-ragknn-{model_id}")
        print(f'endpoint_name={endpoint_name}')

        inference_instance_type = MODEL_CONFIG[model_id]["instance type"]
        print(f'inference_instance_type={inference_instance_type}')

        # Retrieve the inference container uri. This is the base HuggingFace container image for the default model above.
        deploy_image_uri = image_uris.retrieve(
            region=aws_region,
            framework=None,  # automatically inferred from model_id
            image_scope="inference",
            model_id=model_id,
            model_version=model_version,
            instance_type=inference_instance_type,
        )
        print(f'deploy_image_uri={deploy_image_uri}')

        model_uri = model_uris.retrieve(model_id=model_id, model_version=model_version, model_scope="inference")
        print(f'model_uri={model_uri}')

        model_inference = Model(
            image_uri=deploy_image_uri,
            model_data=model_uri,
            role=aws_role,
            predictor_cls=Predictor,
            name=endpoint_name,
            env=MODEL_CONFIG[model_id]["env"],
        )

        model_predictor_inference = model_inference.deploy(
            initial_instance_count=1,
            instance_type=inference_instance_type,
            predictor_cls=Predictor,
            endpoint_name=endpoint_name,  # update_endpoint=True
        )
        MODEL_CONFIG[model_id]["endpoint_name"] = endpoint_name
        print(f'model_predictor_inference={model_predictor_inference}')
        print()
        print(f"Model {model_id} has been deployed successfully.")


def query_endpoint_with_json_payload(encoded_json, endpoint_name, content_type="application/json"):
    client = boto3.client("runtime.sagemaker")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType=content_type, Body=encoded_json
    )
    return response


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


if __name__ == '__main__':
    main()
