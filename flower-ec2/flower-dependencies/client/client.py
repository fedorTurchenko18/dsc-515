import flwr as fl, numpy as np, os, sys
from flwr.common.logger import log
from logging import INFO
curdir = os.path.dirname(__file__)
sys.path.append(os.path.join(curdir, '../../'))
from aws_management.aws_manager import AWSManager
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

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
    def __init__(self, model, train_dataset, test_dataset, instance_id, epochs=1, batch_size=32, cloudwatch_manager=cloudwatch_manager):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.instance_id = instance_id
        self.cloudwatch_manager=cloudwatch_manager


    def get_parameters(self, config):
        '''
        "This method just needs to exist", - official video from docs
        '''
        return [np.asarray(v) for v in self.model.get_weights()]


    def fit(self, parameters, config):
        start_time = datetime.utcnow()
        self.model.set_weights(parameters)
        self.model.fit(self.train_dataset)
        end_time = datetime.utcnow()
        cpu_usage = self.cloudwatch_manager.get_metric_statistic_cloudwatch(instance_id=self.instance_id, start_time=start_time, end_time=end_time)
        log(INFO, f'Instance {self.instance_id} : CPU usage history during fit() : {cpu_usage}')
        return [np.asarray(v) for v in self.model.get_weights()], sum([len(i[1]) for i in self.train_dataset.as_numpy_iterator()]), {}


    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_dataset)
        start_time = datetime.utcnow()
        end_time = datetime.utcnow()
        cpu_usage = self.cloudwatch_manager.get_metric_statistic_cloudwatch(instance_id=self.instance_id, start_time=start_time, end_time=end_time)
        log(INFO, f'Instance {self.instance_id} : CPU usage history during evaluate() : {cpu_usage}')
        return loss, sum([len(i[1]) for i in self.train_dataset.as_numpy_iterator()]), {'accuracy': float(accuracy)}