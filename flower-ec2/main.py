import os, argparse, time
from dotenv import load_dotenv
from aws_management.aws_manager import AWSManager

if __name__=='__main__':

    load_dotenv()
    AWS_ACCESS_KEY = os.environ['AWS_LAB_ACCESS_KEY']
    AWS_SECRET_ACCESS_KEY = os.environ['AWS_LAB_SECRET_ACCESS_KEY']
    AWS_SESSION_TOKEN = os.environ['AWS_LAB_SESSION_TOKEN']
    AWS_REGION = os.environ['AWS_REGION']
    AWS_KEY_PAIR = os.environ['AWS_KEY_PAIR']

    parser = argparse.ArgumentParser(description='Run Flower client with subset of data')
    parser.add_argument('--n_client_instances', default=5, type=int, help='Number of Clients', required=True)
    parser.add_argument('--num_rounds', type=int, default=10, help='Number of Federated Learning rounds to run', required=True)
    parser.add_argument('--strategy', nargs='+', type=str, default=["FedAvg", "FedAvgM", "FedAdaGrad", "FedAdam"], help='Strategy(-ies) to initialize Flower Server with. Available: "FedAvg", "FedAvgM", "FedAdaGrad", "FedAdam"', required=True)
    parser.add_argument('--backend', type=str, help='Backend of Flower Client. Available: "jax", "torch", "tensorflow"', required=True)
    
    args = parser.parse_args()
    N_CLIENT_INSTANCES = args.n_client_instances
    NUM_ROUNDS = args.num_rounds
    STRATEGY = args.strategy
    BACKEND = args.backend

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
    port_22_response = ec2_manager.open_port(default_group=True, port_number=22)

    # create and run the server ec2 instance
    with open('flower-dependencies/server/server_startup.txt', 'r') as f:
        server_startup_script = f.read()

    for to_replace, replacement in {
        'NUM_ROUNDS': str(NUM_ROUNDS),
        'DATA_N': str(N_CLIENT_INSTANCES),
        'STRATEGY': STRATEGY,
        'BACKEND': BACKEND,
        'PASSED_ACCESS_KEY': AWS_ACCESS_KEY,
        'PASSED_SECRET_ACCESS_KEY': AWS_SECRET_ACCESS_KEY,
        'PASSED_SESSION_TOKEN': AWS_SESSION_TOKEN,
        'PASSED_REGION': AWS_REGION,
        'PASSED_KEY_PAIR': AWS_KEY_PAIR
    }.items():
        server_startup_script = server_startup_script.replace(to_replace, replacement)

    flower_server = ec2_manager.create_instance('server', startup_script=server_startup_script, n_instances=1)

    # Extract the public IP address from the response
    response = ec2_manager.client.describe_instances(InstanceIds=[flower_server[0].instance_id])
    server_public_ip_address = response['Reservations'][0]['Instances'][0]['PublicIpAddress']

    # create and run the clients ec2 instances
    with open('flower-dependencies/client/client_startup.txt', 'r') as f:
        client_startup_script = f.read()

    for to_replace, replacement in {
        'PUBLIC_IP': server_public_ip_address,
        'DATA_N': str(N_CLIENT_INSTANCES),
        'BACKEND': BACKEND,
        'STRATEGY': STRATEGY,
        'PASSED_ACCESS_KEY': AWS_ACCESS_KEY,
        'PASSED_SECRET_ACCESS_KEY': AWS_SECRET_ACCESS_KEY,
        'PASSED_SESSION_TOKEN': AWS_SESSION_TOKEN,
        'PASSED_REGION': AWS_REGION,
        'PASSED_KEY_PAIR': AWS_KEY_PAIR
    }.items():
        client_startup_script = client_startup_script.replace(to_replace, replacement)

    flower_clients = []
    for idx in range(N_CLIENT_INSTANCES):
        client_specific_startup_script = client_startup_script.replace('DATA_INDEX', str(idx))
        flower_client = ec2_manager.create_instance('clients', startup_script=client_specific_startup_script, n_instances=1)
        flower_clients.append(flower_client[0].instance_id)

    s3_manager = AWSManager(
        service='s3',
        aws_access_key=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        aws_region=AWS_REGION,
        aws_key_pair=AWS_KEY_PAIR
    )
    local_log_path = os.path.abspath(__file__)
    local_log_path = local_log_path[:local_log_path.rindex('/')]
    server_local_log_path = f'{local_log_path}/server_log.log'
    client_local_log_path = f'{local_log_path}/client_log.log'

    out_cond = False
    retry_attempts = 0
    while out_cond == False and retry_attempts < 20:
        fl_server_log_download_response = s3_manager.download_from_s3_bucket(local_file_path=server_local_log_path, object_key=f'{BACKEND}/{STRATEGY}/server_log.log')
        fl_client_log_download_response = s3_manager.download_from_s3_bucket(local_file_path=client_local_log_path, object_key=f'{BACKEND}/{STRATEGY}/client_log.log')
        try:
            if 'error' in fl_server_log_download_response or 'error' in fl_client_log_download_response:
                out_cond = False
                retry_attempts += 1
                time.sleep(30)
        except TypeError:
            out_cond = True
    else:
        print('Falied to fetch the logs from s3')

    # clean-up
    ec2_manager.terminate_instance([flower_server[0].instance_id]+[i.instance_id for i in flower_clients])
    if os.path.isfile(server_local_log_path) and os.path.isfile(server_local_log_path):
        s3_manager.delete_s3_bucket()
    else:
        print('Cannot delete bucket as the files were not donwloaded. You have to do it manually from AWS UI')