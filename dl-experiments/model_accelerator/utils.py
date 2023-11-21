from tensorflow import keras

def functional_model_from_config(config):
    layers = config['layers']
    deserialized_layers = [keras.layers.deserialize(layer) for layer in layers]

    input_layer = deserialized_layers[0]
    outputs = input_layer.output
    for dszd_layer in deserialized_layers:
        outputs = dszd_layer(outputs)
    inputs = input_layer.input

    return inputs, outputs