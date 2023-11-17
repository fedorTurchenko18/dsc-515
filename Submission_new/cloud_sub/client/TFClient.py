import flwr as fl

import flwr as fl
import numpy as np
import tensorflow as tf

# Define Flower client for TensorFlow
class TensorFlowClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, test_dataset, epochs=1, batch_size=32):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["categorical_accuracy"])

    def get_parameters(self):
        # Convert model parameters to a list of NumPy ndarrays
        return [np.asarray(v) for v in self.model.get_weights()]

    def set_parameters(self, parameters):
        # Set model parameters from a list of NumPy ndarrays
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        # Set the parameters, train the model, return the updated parameters
        self.set_parameters(parameters)
        self.model.fit(self.train_dataset, epochs=self.epochs)
        return self.get_parameters(), sum(1 for _ in self.train_dataset), {}

    def evaluate(self, parameters, config):
        # Set the parameters, evaluate the model, return the result
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.test_dataset)
        return float(loss), sum(1 for _ in self.test_dataset), {"accuracy": float(accuracy)}

def split_dataset(dataset, n_splits, index):
    """
    Splits the dataset into n_splits parts and returns the part at the specified index.
    """
    dataset_size = sum(1 for _ in dataset)
    split_size = dataset_size // n_splits
    start = split_size * index
    end = start + split_size if index < n_splits - 1 else dataset_size
    return dataset.take(end).skip(start)

    
    
import os, numpy as np, tensorflow as tf
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run Flower client with subset of data')
    parser.add_argument('--data_index', type=int, help='Index of data subset to use', required=False)
    parser.add_argument('--data_n', type=int, help='Number of Clients', required=False)

    args = parser.parse_args()

    data_index = args.data_index

    data_n = args.data_n
    
    DIR = '../images_houseware'
    VAL_SHARE = 0.25
    LABEL_MODE = 'categorical'
    SEED = 515
    SUBSET = 'both'
    BATCH_SIZE = 32


    train, test = tf.keras.utils.image_dataset_from_directory(
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
    
    classes_num = 10

    image_dims = (256, 256, 3)

    tf_model = tf.keras.applications.VGG16(input_shape=image_dims, classes=classes_num,include_top=False)
    client = TensorFlowClient(tf_model, train, test , epochs =1 , batch_size=32)

    fl.client.start_numpy_client(
        server_address='server:8080',  # Adjusted server address
        client=client
    )