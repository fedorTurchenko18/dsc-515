import keras_core as keras, os, pickle, flwr as fl, numpy as np
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()
HGF_TOKEN = os.environ['HUGGINGFACE_TOKEN']
HGF_DATA_REPO = os.environ['HUGGINGFACE_DATASET_V2_REPO']
HGF_TOPIC_MODEL_TOP2VEC_REPO = os.environ['HUGGINGFACE_TOPIC_MODEL_TOP2VEC_REPO']
HGF_TOPIC_MODEL_TOP2VEC_FILE = os.environ['HUGGINGFACE_TOPIC_MODEL_TOP2VEC_FILE']
OPTIMIZER = keras.optimizers.Adam(learning_rate=0.001)
LOSS = 'categorical_crossentropy'
METRICS = [keras.metrics.CategoricalAccuracy()]
N_CLASSES = 10

inputs = keras.Input(shape=(256, 256, 3))
conv1 = keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
maxpool1 = keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(conv1)
conv2 = keras.layers.Conv2D(32, (3, 3), activation='relu')(maxpool1)
maxpool2 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)
conv3 = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool2)
maxpool3 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)
flat = keras.layers.Flatten()(maxpool3)
dense = keras.layers.Dense(128, activation='relu')(flat)
outputs = keras.layers.Dense(N_CLASSES, activation='softmax')(dense)
model = keras.Model(inputs=inputs, outputs=outputs)


class UniversalClient(fl.client.NumPyClient):
    def __init__(self, train_dataset, test_dataset, model=model, epochs=1, batch_size=32):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.model.compile(
            optimizer=OPTIMIZER,
            loss=LOSS,
            metrics=METRICS
        )


    def get_parameters(self, config):
        '''
        "This method just needs to exist", - official video from docs
        '''
        return self.model.get_weights()


    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train_dataset)
        return self.model.get_weights(), sum([len(i[1]) for i in self.train_dataset.as_numpy_iterator()]), {}


    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_dataset)
        return loss, sum([len(i[1]) for i in self.train_dataset.as_numpy_iterator()]), {'accuracy': float(accuracy)}