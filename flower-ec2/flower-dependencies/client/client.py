import flwr as fl, numpy as np

class UniversalClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, test_dataset, epochs=1, batch_size=32):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.batch_size = batch_size


    def get_parameters(self, config):
        '''
        "This method just needs to exist", - official video from docs
        '''
        return [np.asarray(v) for v in self.model.get_weights()]


    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train_dataset)
        return [np.asarray(v) for v in self.model.get_weights()], sum([len(i[1]) for i in self.train_dataset.as_numpy_iterator()]), {}


    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_dataset)
        return loss, sum([len(i[1]) for i in self.train_dataset.as_numpy_iterator()]), {'accuracy': float(accuracy)}