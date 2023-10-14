import sagemaker, boto3, json
from sagemaker.session import Session
from sagemaker.model import Model
from sagemaker import image_uris, model_uris, script_uris, hyperparameters
from sagemaker.predictor import Predictor
from sagemaker.utils import name_from_base

from llmrag.config import MODEL_CONFIG


def query_endpoint_with_json_payload(encoded_json, endpoint_name, content_type="application/json"):
    client = boto3.client("runtime.sagemaker")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType=content_type, Body=encoded_json
    )
    return response


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


if __name__ == '__main__':
    main()
