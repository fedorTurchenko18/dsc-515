import os, argparse
from dotenv import load_dotenv
from aws_management.aws_manager import AWSManager
from loguru import logger

if __name__=='__main__':

    load_dotenv()
    AWS_ACCESS_KEY = os.environ['AWS_LAB_ACCESS_KEY']
    AWS_SECRET_ACCESS_KEY = os.environ['AWS_LAB_SECRET_ACCESS_KEY']
    AWS_SESSION_TOKEN = os.environ['AWS_LAB_SESSION_TOKEN']
    AWS_REGION = os.environ['AWS_REGION']
    AWS_KEY_PAIR = os.environ['AWS_KEY_PAIR']

    parser = argparse.ArgumentParser(description='Run Flower client with subset of data')
    parser.add_argument('--n_client_instances', type=int, help='Number of Clients', required=True)
    args = parser.parse_args()
    N_CLIENT_INSTANCES = args.n_client_instances

    ec2_manager = AWSManager(
        service='ec2',
        aws_access_key=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        aws_region=AWS_REGION,
        aws_key_pair=AWS_KEY_PAIR
    )

    # add rule to the default security group to open port 8080 and 22
    port_8080_response = ec2_manager.open_port(default_group=True, port_number=8080)
    logger.info(f'Response of opening port 8080 operation: {port_8080_response}')
    port_22_response = ec2_manager.open_port(default_group=True, port_number=22)
    logger.info(f'Response of opening port 22 operation: {port_22_response}')

    # create s3 bucket for Flower logs storage
    # s3_manager = AWSManager(
    #     service='s3',
    #     aws_access_key=AWS_ACCESS_KEY,
    #     aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    #     aws_session_token=AWS_SESSION_TOKEN,
    #     aws_region=AWS_REGION,
    #     aws_key_pair=AWS_KEY_PAIR
    # )
    # s3_response = s3_manager.create_s3_bucket()

    # create and run the server ec2 instance
    with open('flower-dependencies/server/server_startup.txt', 'r') as f:
        server_startup_script = f.read()
    server_startup_script = server_startup_script.replace('DATA_N', str(N_CLIENT_INSTANCES))
    logger.info('Creating Server instance...')
    flower_server = ec2_manager.create_instance('server', startup_script=server_startup_script, n_instances=1)

    # Extract the public IP address from the response
    response = ec2_manager.client.describe_instances(InstanceIds=[flower_server[0].instance_id])
    server_public_ip_address = response['Reservations'][0]['Instances'][0]['PublicIpAddress']

    # create and run the clients ec2 instances
    with open('flower-dependencies/client/client_startup.txt', 'r') as f:
        client_startup_script = f.read()

    for to_replace, replacement in {'EC2_PUBLIC_IP': server_public_ip_address, 'DATA_N': str(N_CLIENT_INSTANCES)}.items():
        client_startup_script = client_startup_script.replace(to_replace, replacement)

    logger.info(f'Received request to create {N_CLIENT_INSTANCES} Client instances')
    for idx in range(N_CLIENT_INSTANCES):
        client_specific_startup_script = client_startup_script.replace('DATA_INDEX', str(idx))
        logger.info(f'Creating Client #{idx+1} instance...')
        flower_clients = ec2_manager.create_instance('clients', startup_script=client_specific_startup_script, n_instances=1)