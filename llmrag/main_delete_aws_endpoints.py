import boto3


def main():
    sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
    endpoints = sagemaker_client.list_endpoints(MaxResults=100)['Endpoints']
    for endpoint in endpoints:
        print(f'{endpoint["EndpointStatus"]} -- {endpoint}')

        # Uncomment to actually delete:
        # try:
        #     endpoint_name = endpoint['EndpointName']
        #     print(f'Deleting {endpoint_name}...')
        #     sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        #     print(f'{endpoint_name} deleted.')
        # except Exception as e:
        #     print(e)


if __name__ == '__main__':
    main()
