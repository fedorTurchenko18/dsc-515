import os
import flwr as fl
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras

class IrisClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test, model, custom_config):
        super().__init__()
        self.model = model
        self.custom_config = custom_config
        self.parameters = model.get_weights()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def get_parameters(self, config):
        '''
        "This method just needs to exist", - official video from docs
        '''
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.y_train, **self.custom_config)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        return loss, len(self.X_test), {'accuracy': float(accuracy)}
    
if __name__=='__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1,1)).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    keras.utils.set_random_seed(515)
    tf.config.experimental.enable_op_determinism()

    # Define training parameters
    EPOCHS = 20
    VALIDATION_SPLIT = 0.1
    NEURONS = 64
    ACTIVATION_FUNCTION = 'relu'
    BATCH_SIZE = 8
    LEARNING_RATE = 0.03
    MOMENTUM = 0.9

    model_training_params = {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'validation_split': VALIDATION_SPLIT,
        'verbose': False
    }

    model = keras.Sequential(
        [
            keras.layers.Dense(NEURONS, activation=ACTIVATION_FUNCTION, input_shape=(X_train.shape[1],)),
            keras.layers.Dense(3, activation='softmax')
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    fl.client.start_numpy_client(
        server_address='[::]:8080',
        client=IrisClient(
            model=model,
            custom_config=model_training_params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
    )