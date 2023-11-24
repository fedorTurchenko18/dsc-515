import sys, os, argparse, flwr as fl
curdir = os.path.dirname(__file__)
sys.path.append(os.path.join(curdir, '../../'))
from aws_management.aws_manager import AWSManager

if __name__ == '__main__':

    AWS_ACCESS_KEY = os.environ['AWS_LAB_ACCESS_KEY']
    AWS_SECRET_ACCESS_KEY = os.environ['AWS_LAB_SECRET_ACCESS_KEY']
    AWS_SESSION_TOKEN = os.environ['AWS_LAB_SESSION_TOKEN']
    AWS_REGION = os.environ['AWS_REGION']
    AWS_KEY_PAIR = os.environ['AWS_KEY_PAIR']

    cur_abs_path = os.path.abspath('.')
    DIR = f"{cur_abs_path[:cur_abs_path.find('dsc-515')+len('dsc-515')]}/images_houseware"
    VAL_SHARE = 0.25
    LABEL_MODE = 'categorical'
    SEED = 515
    SUBSET = 'both'
    BATCH_SIZE = 32

    def split_dataset(dataset, n_splits, index):
        """
        Splits the dataset into n_splits parts and returns the part at the specified index.
        """
        dataset_size = sum(1 for _ in dataset)
        split_size = dataset_size // n_splits
        start = split_size * index
        end = start + split_size if index < n_splits - 1 else dataset_size
        return dataset.take(end).skip(start)

    parser = argparse.ArgumentParser(description='Run Flower client with subset of data')
    parser.add_argument('--backend', type=str, help='Backend of Flower Client. Available: "jax", "torch", "tensorflow"', required=True)
    parser.add_argument('--data_index', type=int, help='Index of data subset to use', required=True)
    parser.add_argument('--data_n', type=int, help='Number of Clients', required=True)
    parser.add_argument('--public_ip', type=str, help='Public IP address of the Server instance', required=True)
    parser.add_argument('--instance_id', type=str, help='ID of the Server instance (required for CPU usage measurement)', required=True)
    parser.add_argument('--strategy', type=str, help='Strategy to initialize Flower Server with (needed here to compose the log file path on s3). Available: "FedAvg", "FedAvgM", "FedAdaGrad", "FedAdam"', required=True)

    args = parser.parse_args()

    backend = args.backend
    os.environ['KERAS_BACKEND']=backend
    # moving keras_core related imports here
    # because KERAS_BACKEND env variable must be declared
    # before it is imported
    import keras_core as keras
    from client import UniversalClient
    curdir = os.path.dirname(__file__)
    sys.path.append(os.path.join(curdir, '..'))
    from model import KerasCoreCNN

    data_index = args.data_index
    data_n = args.data_n
    server_public_ip = args.public_ip
    instance_id = args.instance_id
    strategy_str = args.strategy

    train, test = keras.utils.image_dataset_from_directory(
        DIR,
        label_mode=LABEL_MODE,
        validation_split=VAL_SHARE,
        seed=SEED,
        subset=SUBSET,
        batch_size=BATCH_SIZE
    )

    train = train.map(lambda image, label: (image / 255.0, label))
    test = test.map(lambda image, label: (image / 255.0, label))
    train = split_dataset(train, data_n, data_index)
    
    model = KerasCoreCNN().model

    client = UniversalClient(model, train, test, instance_id)

    s3_manager = AWSManager(
        service='s3',
        aws_access_key=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        aws_region=AWS_REGION,
        aws_key_pair=AWS_KEY_PAIR
    )
    log_dir = os.path.abspath(__file__)
    log_dir = log_dir[:log_dir.rindex('/')]
    log_file = f'{log_dir}/client_log.txt'

    fl.common.logger.configure(identifier=f'{backend}-{strategy_str}-run.txt', filename=log_file)

    fl.client.start_numpy_client(
        server_address=f'{server_public_ip}:8080',
        client=client
    )

    # save FL log to s3
    write_to_s3_bucket_response = s3_manager.write_to_s3_bucket(log_file=log_file, object_key=f'{backend}/{strategy_str}/client_log.txt')