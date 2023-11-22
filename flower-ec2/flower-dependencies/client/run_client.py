import os, argparse, flwr as fl
os.environ['KERAS_BACKEND']='jax'
import keras_core as keras
from client import UniversalClient

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Flower client with subset of data')
    parser.add_argument('--data_index', type=int, help='Index of data subset to use', required=False)
    parser.add_argument('--data_n', type=int, help='Number of Clients', required=False)

    args = parser.parse_args()
    data_index = args.data_index
    data_n = args.data_n

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

    client = UniversalClient(train, test)

    fl.client.start_numpy_client(
        server_address='[::]:8080',
        client=client
    )