import boto3
import yaml
from typing import Literal, List

with open('ec2_management/ec2_config.yaml', 'rb') as f:
    ec2_config = yaml.safe_load(f)

CLIENT_IMAGE_ID = ec2_config['clients']['ImageId']
CLIENT_MIN_COUNT = ec2_config['clients']['MinCount']
CLIENT_MAX_COUNT = ec2_config['clients']['MaxCount']
CLIENT_INSTANCE_TYPE = ec2_config['clients']['InstanceType']

SERVER_IMAGE_ID = ec2_config['server']['ImageId']
SERVER_MIN_COUNT = ec2_config['server']['MinCount']
SERVER_MAX_COUNT = ec2_config['server']['MaxCount']
SERVER_INSTANCE_TYPE = ec2_config['server']['InstanceType']

class EC2Client:
    '''
    Class for managing of ec2 instances
    '''
    def __init__(self, aws_access_key: str, aws_secret_access_key: str, aws_session_token: str, aws_region: Literal['us-east-1', 'us-west-2'], aws_key_pair: str) -> None:
        '''
        Parameters
        ----------

        aws_access_key : str
            Access key from AWS credentials

        aws_secret_access_key : str
            Secret access key from AWS credentials

        aws_region : str
            AWS region (only `'us-east-1'` or `'us-west-2'` are allowed)

        key_pair : str
            AWS .pem file name
        '''
        self.__key_pair = aws_key_pair
        self.client = boto3.client(
            'ec2',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=aws_region
        )


    def create_instance(self, instance_type: Literal['server', 'clients']):
        '''
        Create ec2 instance

        Parameters
        ----------

        instance_type : Literal['server', 'clients']
            Type of instance, respective to Flower's client-server paradigm

        Returns
        -------

        List of `self.client.Instance`
        '''
        if instance_type=='clients':
            instance = self.client.run_instances(
                ImageId=CLIENT_IMAGE_ID,
                MinCount=CLIENT_MIN_COUNT,
                MaxCount=CLIENT_MAX_COUNT,
                InstanceType=CLIENT_INSTANCE_TYPE,
                KeyName=self.__key_pair
            )
        elif instance_type=='server':
            instance = self.client.run_instances(
                ImageId=SERVER_IMAGE_ID,
                MinCount=SERVER_MIN_COUNT,
                MaxCount=SERVER_MAX_COUNT,
                InstanceType=SERVER_INSTANCE_TYPE,
                KeyName=self.__key_pair
            )

        return instance
    

    def terminate_instance(self, instance_ids: List[str]) -> dict:
        '''
        Terminate ec2 instances

        Parameters
        ----------

        instance_ids : List[str]
            List of identifiers for instances to terminate

        Returns
        -------

        Operation response in a JSON format
        '''
        response = self.client.terminate_instances(InstanceIds=instance_ids)
        return response

class EC2Instance(EC2Client):
    '''
    Class for accessing and performing actions on indivdual ec2 instance
    '''
    def __init__(self, aws_access_key: str, aws_secret_access_key: str, aws_region: Literal['us-east-1', 'us-west-2'], key_pair: str) -> None:
        super().__init__(aws_access_key, aws_secret_access_key, aws_region, key_pair)