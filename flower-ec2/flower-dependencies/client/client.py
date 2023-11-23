import flwr as fl, numpy as np, os, sys
from flwr.common.logger import log
from logging import INFO
curdir = os.path.dirname(__file__)
sys.path.append(os.path.join(curdir, '../../'))
from aws_management.aws_manager import AWSManager
from datetime import datetime
from typing import Literal, List

AWS_ACCESS_KEY = os.environ['AWS_LAB_ACCESS_KEY']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_LAB_SECRET_ACCESS_KEY']
AWS_SESSION_TOKEN = os.environ['AWS_LAB_SESSION_TOKEN']
AWS_REGION = os.environ['AWS_REGION']
AWS_KEY_PAIR = os.environ['AWS_KEY_PAIR']

cloudwatch_manager = AWSManager(
    service='cloudwatch',
    aws_access_key=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
    aws_region=AWS_REGION,
    aws_key_pair=AWS_KEY_PAIR
)

class UniversalClient(fl.client.NumPyClient):
    def __init__(
            self,
            model: Literal['keras_core.src.models.functional.Functional'],
            train_dataset: List[Literal['tf.data.Dataset']],
            test_dataset: List[Literal['tf.data.Dataset']],
            instance_id: str,
            epochs=1, batch_size=32, cloudwatch_manager: AWSManager = cloudwatch_manager
    ):
        '''
        Initialize universal Flower client. 
        Universal means that it is based on `keras_core` library methods only
        which allow it to run on different backends, namely JAX/Tensorflow/Pytorch

        Parameters
        ----------

        model : Literal['keras_core.src.models.functional.Functional']
            Compiled `keras_core` Functional model

        train_dataset : Literal['tf.data.Dataset']
            Train data created using `keras_core.datasets.image_dataset_from_directory`

        test_dataset : Literal['tf.data.Dataset']
            Test data created using `keras_core.datasets.image_dataset_from_directory`

        instance_id : str
            ID of the Flower Client instance on which this client will be running \n
            Needed for CPU usage monitoring during `fit()` and `evaluate()`

        epochs : int
            How many epochs to train model for

        batch_size : int
            Batch size for model training

        cloudwatch_manager : AWSManager
            Custom class `AWSManager` initialized with `service='cloudwatch'` argument \n
            Needed for CPU usage monitoring during `fit()` and `evaluate()`
        '''
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.instance_id = instance_id
        self.cloudwatch_manager=cloudwatch_manager


    def get_parameters(self, config):
        '''
        Implementation of this method follows official Flower documentation
        '''
        return [np.asarray(v) for v in self.model.get_weights()]


    def fit(self, parameters, config):
        '''
        Implementation of this method follows official Flower documentation \n
        CPU usage monitoring is the only custom implemented functionality
        '''
        self.model.set_weights(parameters)
        # start point of CPU usage monitoring
        start_time = datetime.utcnow()
        self.model.fit(self.train_dataset)
        # end point of CPU usage monitoring
        end_time = datetime.utcnow()
        # measure CPU usage monitoring
        cpu_usage = self.cloudwatch_manager.get_metric_statistic_cloudwatch(instance_id=self.instance_id, start_time=start_time, end_time=end_time)
        # log CPU usage monitoring
        log(INFO, f'Instance {self.instance_id} : CPU usage history during fit() : {cpu_usage}')
        return [np.asarray(v) for v in self.model.get_weights()], sum([len(i[1]) for i in self.train_dataset.as_numpy_iterator()]), {}


    def evaluate(self, parameters, config):
        '''
        Implementation of this method follows official Flower documentation \n
        CPU usage monitoring is the only custom implemented functionality
        '''
        self.model.set_weights(parameters)
        # start point of CPU usage monitoring
        start_time = datetime.utcnow()
        loss, accuracy = self.model.evaluate(self.test_dataset)
        # end point of CPU usage monitoring
        end_time = datetime.utcnow()
        # measure CPU usage monitoring
        cpu_usage = self.cloudwatch_manager.get_metric_statistic_cloudwatch(instance_id=self.instance_id, start_time=start_time, end_time=end_time)
        # log CPU usage monitoring
        log(INFO, f'Instance {self.instance_id} : CPU usage history during evaluate() : {cpu_usage}')
        return loss, sum([len(i[1]) for i in self.train_dataset.as_numpy_iterator()]), {'accuracy': float(accuracy)}