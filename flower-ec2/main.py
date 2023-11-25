import os, argparse, time
from dotenv import load_dotenv
from aws_management.aws_manager import AWSManager
from loguru import logger

if __name__=='__main__':
    start = time.time()
    load_dotenv()
    AWS_ACCESS_KEY = os.environ['AWS_LAB_ACCESS_KEY']
    AWS_SECRET_ACCESS_KEY = os.environ['AWS_LAB_SECRET_ACCESS_KEY']
    AWS_SESSION_TOKEN = os.environ['AWS_LAB_SESSION_TOKEN']
    AWS_REGION = os.environ['AWS_REGION']
    AWS_KEY_PAIR = os.environ['AWS_KEY_PAIR']

    parser = argparse.ArgumentParser(description='Run Flower client with subset of data')
    parser.add_argument('--n_client_instances', default=5, type=int, help='Number of Clients', required=False)
    parser.add_argument('--num_rounds', type=int, default=10, help='Number of Federated Learning rounds to run', required=False)
    parser.add_argument('--strategy', nargs='+', type=str, default=["FedAvg", "FedAvgM", "FedAdaGrad", "FedAdam"], help='Strategy(-ies) to initialize Flower Server with. Available: "FedAvg", "FedAvgM", "FedAdaGrad", "FedAdam"', required=False)
    parser.add_argument('--backend', type=str, help='Backend of Flower Client. Available: "jax", "torch", "tensorflow"', required=False)
    
    args = parser.parse_args()
    N_CLIENT_INSTANCES = args.n_client_instances
    NUM_ROUNDS = args.num_rounds
    STRATEGY = ' '.join(args.strategy)
    BACKEND = args.backend

    logger.info(f"Starting workflow with parameters:\nNumber of Clients: {N_CLIENT_INSTANCES}\nClients' backend: {BACKEND}\nServer strategy: {STRATEGY}\nNumber of FL rounds: {NUM_ROUNDS}")

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

    logger.info('Starting server...')
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
        logger.info(f'Starting #{idx+1} Client...')
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

    local_client_results_paths = []
    local_server_results_paths = []
    for strategy in STRATEGY.split(' '):
        local_log_path = os.path.abspath(__file__)
        local_log_path = local_log_path[:local_log_path.rindex('/')]
        os.makedirs(f'{local_log_path}/federated-learning-results/{BACKEND}/{strategy}', exist_ok=True)
        server_local_log_path = f'{local_log_path}/federated-learning-results/{BACKEND}/{strategy}/server_log.log'
        client_local_log_path = f'{local_log_path}/federated-learning-results/{BACKEND}/{strategy}/client_log.log'

        local_server_results_paths.append(server_local_log_path)
        local_client_results_paths.append(client_local_log_path)

        out_cond = False
        retry_attempts = 0
        while out_cond == False and retry_attempts < 60:
            logger.info(f'Attempt #{retry_attempts+1} of 60 | Waiting for the server results from strategy {strategy}')
            fl_server_log_download_response = s3_manager.download_from_s3_bucket(local_file_path=server_local_log_path, object_key=f'{BACKEND}/{strategy}/server_log.log')
            fl_client_log_download_response = s3_manager.download_from_s3_bucket(local_file_path=client_local_log_path, object_key=f'{BACKEND}/{strategy}/client_log.log')
            try:
                if 'error' in fl_server_log_download_response or 'error' in fl_client_log_download_response:
                    out_cond = False
                    retry_attempts += 1
                    time.sleep(60)
            except TypeError:
                logger.info(f'Successfully fetched logs for {strategy} from s3. Waiting for the next strategy results...')
                out_cond = True
        else:
            logger.info(f'Falied to fetch the logs for {strategy} from s3. Waiting for the next strategy results...')
    end = time.time()
    logger.info(f'Workflow took {end-start} seconds')
    
    ## clean-up
    # terminate ec2 instances
    logger.info('Terminating ec2 instances...')
    ec2_manager.terminate_instance([flower_server[0].instance_id]+[i for i in flower_clients])
    logger.info('Done')
    
    # empty and delete s3 bucket
    for s_p, c_p in zip(local_server_results_paths, local_client_results_paths):
        checks = set()
        if os.path.isfile(s_p) and os.path.isfile(c_p):
            checks.add(True)
        else:
            checks.add(False)
    if len(checks) != 1:
        logger.info('Cannot delete bucket as som of the files were not donwloaded. You have to do it manually from AWS UI')
    else:
        s3_manager.delete_s3_bucket()