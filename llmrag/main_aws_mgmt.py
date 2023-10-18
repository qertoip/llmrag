import boto3


def print_endpoints():
    sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
    endpoints = sagemaker_client.list_endpoints(MaxResults=100)['Endpoints']
    for endpoint in endpoints:
        print(f'{endpoint["EndpointStatus"]} -- {endpoint}')


def delete_endpoints():
    sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
    endpoints = sagemaker_client.list_endpoints(MaxResults=100)['Endpoints']
    for endpoint in endpoints:
        try:
            endpoint_name = endpoint['EndpointName']
            print(f'Deleting {endpoint_name}...')
            # Uncomment to actually delete:
            # sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            print(f'{endpoint_name} deleted.')
        except Exception as e:
            print(e)

def print_models():
    sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
    response = sagemaker_client.list_models()
    for model in response['Models']:
        model_name = model['ModelName']
        print(model_name)


def delete_models():
    sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
    response = sagemaker_client.list_models()
    for model in response['Models']:
        model_name = model['ModelName']
        print(f"Deleting model: {model_name}")
        sagemaker_client.delete_model(ModelName=model_name)


if __name__ == '__main__':
    #print_endpoints()
    #delete_endpoints()

    print_models()
    delete_models()
