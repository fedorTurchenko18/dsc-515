import flwr as fl, argparse, os, numpy as np, sys
import keras_core as keras
from strategy import ServerStrategy
curdir = os.path.dirname(__file__)
sys.path.append(os.path.join(curdir, '..'))
from model import KerasCoreCNN


if __name__=='__main__':
    cur_abs_path = os.path.abspath('.')
    DIR = f"{cur_abs_path[:cur_abs_path.find('dsc-515')+len('dsc-515')]}/images_houseware"
    VAL_SHARE = 0.25
    LABEL_MODE = 'categorical'
    SEED = 515
    SUBSET = 'both'
    BATCH_SIZE = 32

    parser = argparse.ArgumentParser(description='Run Flower client with subset of data')
    parser.add_argument('--data_n', type=int, help='Number of Clients', required=True)

    args = parser.parse_args()
    data_n = args.data_n

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
    
    strat_wrapper = ServerStrategy(
        fl_strategy=fl.server.strategy.FedAvg,
        min_available_clients=data_n,
        min_fit_clients=data_n,
        min_evaluate_clients=data_n,
        evaluate_fn=evaluate,
        initial_parameters=initial_parameters
    )
    
    fl.server.start_server(
        server_address='0.0.0.0:8080',
        strategy=strat_wrapper.strategy,
        config=fl.server.ServerConfig(num_rounds=3)
    ) 