import flwr as fl, argparse, os, numpy as np, sys
import keras_core as keras
from strategy import ServerStrategy
curdir = os.path.dirname(__file__)
sys.path.append(os.path.join(curdir, '..'))
from model import KerasCoreCNN
sys.path.append(os.path.join(curdir, '../../'))
from aws_management.aws_manager import AWSManager
from dotenv import load_dotenv
load_dotenv()
from loguru import logger

if __name__=='__main__':
    cur_abs_path = os.path.abspath('.')
    DIR = f"{cur_abs_path[:cur_abs_path.find('dsc-515')+len('dsc-515')]}/images_houseware"
    VAL_SHARE = 0.25
    LABEL_MODE = 'categorical'
    SEED = 515
    SUBSET = 'both'
    BATCH_SIZE = 32

    AWS_ACCESS_KEY = os.environ['AWS_LAB_ACCESS_KEY']
    AWS_SECRET_ACCESS_KEY = os.environ['AWS_LAB_SECRET_ACCESS_KEY']
    AWS_SESSION_TOKEN = os.environ['AWS_LAB_SESSION_TOKEN']
    AWS_REGION = os.environ['AWS_REGION']
    AWS_KEY_PAIR = os.environ['AWS_KEY_PAIR']

    # create s3 bucket for Flower logs storage
    s3_manager = AWSManager(
        service='s3',
        aws_access_key=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        aws_region=AWS_REGION,
        aws_key_pair=AWS_KEY_PAIR
    )
    s3_create_bucket_response = s3_manager.create_s3_bucket()

    parser = argparse.ArgumentParser(description='Run Flower Server')
    parser.add_argument('--num_rounds', type=int, help='Number of Federated Learning rounds to run', required=True)
    parser.add_argument('--data_n', type=int, help='Number of Clients to wait for', required=True)
    parser.add_argument('--strategy', nargs='+', type=str, help='Strategy to initialize Flower Server with. Available: "FedAvg", "FedAvgM", "FedAdaGrad", "FedAdam"', required=True)
    parser.add_argument('--backend', type=str, help='Backend of Flower Client (needed here to compose the log file path on s3). Available: "jax", "torch", "tensorflow"', required=True)

    args = parser.parse_args()
    data_n = args.data_n
    strategies = args.strategy
    backend = args.backend
    num_rounds = args.num_rounds

    _, test = keras.utils.image_dataset_from_directory(
        DIR,
        label_mode=LABEL_MODE,
        validation_split=VAL_SHARE,
        seed=SEED,
        subset=SUBSET,
        batch_size=BATCH_SIZE
    )

    test = test.map(lambda image, label: (image / 255.0, label))

    model = KerasCoreCNN().model
    initial_parameters = fl.common.ndarrays_to_parameters([np.asarray(v) for v in model.get_weights()])

    def evaluate(self, parameters, config):
        model = KerasCoreCNN().model
        model.set_weights(parameters)
        test_dataset = test
        loss, accuracy = model.evaluate(test_dataset)
        return loss, {'accuracy': float(accuracy)}
    
    strategy_mapping = {
        'FedAvg': fl.server.strategy.FedAvg,
        'FedAvgM': fl.server.strategy.FedAvgM,
        'FedAdaGrad': fl.server.strategy.FedAdagrad,
        'FedAdam': fl.server.strategy.FedAdam
    }

    log_dir = os.path.abspath(__file__)
    log_dir = log_dir[:log_dir.rindex('/')]

    for strategy_str in strategies:
        strategy = strategy_mapping[strategy_str]
        strat_wrapper = ServerStrategy(
            fl_strategy=strategy,
            min_available_clients=data_n,
            min_fit_clients=data_n,
            min_evaluate_clients=data_n,
            evaluate_fn=evaluate,
            initial_parameters=initial_parameters
        )

        log_file = f'{log_dir}/{backend}_{strategy_str}_server_log.log'
        fl.common.logger.configure(identifier=f'{backend}-{strategy_str}-run', filename=log_file)

        logger.info(f'Starting server with {strategy_str} strategy')
        fl.server.start_server(
            server_address='0.0.0.0:8080',
            strategy=strat_wrapper.strategy,
            config=fl.server.ServerConfig(num_rounds=num_rounds)
        )

        logger.info(f'Writing {log_file} to s3')
        # save FL log to s3
        write_to_s3_bucket_response = s3_manager.write_to_s3_bucket(log_file=log_file, object_key=f'{backend}/{strategy_str}/server_log.log')
