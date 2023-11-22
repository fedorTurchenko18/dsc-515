import keras_core as keras

INPUT_SHAPE = (256, 256, 3)
N_CLASSES = 10
OPTIMIZER = keras.optimizers.Adam(learning_rate=0.001)
LOSS = 'categorical_crossentropy'
METRICS = [keras.metrics.CategoricalAccuracy()]

class KerasCoreCNN:
    def __init__(self, input_shape=INPUT_SHAPE, optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS, n_classes=N_CLASSES) -> None:
        self.n_classes = n_classes
        self.model = self.__build_model(input_shape)
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def __build_model(self, input_shape):
        inputs = keras.Input(shape=input_shape)
        conv1 = keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
        maxpool1 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)
        conv2 = keras.layers.Conv2D(32, (3, 3), activation='relu')(maxpool1)
        maxpool2 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)
        conv3 = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool2)
        maxpool3 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)
        flat = keras.layers.Flatten()(maxpool3)
        dense = keras.layers.Dense(128, activation='relu')(flat)
        outputs = keras.layers.Dense(self.n_classes, activation='softmax')(dense)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
