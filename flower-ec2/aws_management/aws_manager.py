import boto3, yaml
from botocore.exceptions import ClientError
from typing import Literal, List
from loguru import logger

with open('aws_management/ec2_config.yaml', 'rb') as f:
    ec2_config = yaml.safe_load(f)


class AWSManager:
    '''
    Class for managing of ec2 instances
    '''
    def __init__(self, service: Literal['ec2', 's3'], aws_access_key: str, aws_secret_access_key: str, aws_session_token: str, aws_region: Literal['us-east-1', 'us-west-2'], aws_key_pair: str) -> None:
        '''
        Parameters
        ----------

        service : Literal['ec2', 's3']
            For which service 

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
        self.__region = aws_region
        self.resource = boto3.resource(
            service,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=aws_region
        )
        self.client = boto3.client(
            service,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=aws_region
        )


    def create_instance(self, instance_type: Literal['server', 'clients'], startup_script: str, n_instances: int):
        '''
        Create ec2 instance

        Parameters
        ----------

        instance_type : Literal['server', 'clients']
            Type of instance, respective to Flower's resource-server paradigm

        startup_script : str
            Script to run on ec2 instance start-up. 
            Should be read through context manager from .txt file
        
        n_instances : int
            How many instances to create. Only passed to the routine which creates clients

        Returns
        -------

        List of `self.resource.Instance`
        '''
        if instance_type=='clients':
            instance = self.resource.create_instances(
                MinCount=n_instances,
                MaxCount=n_instances,
                KeyName=self.__key_pair,
                UserData=startup_script,
                **ec2_config['clients']
            )
            logger.info('Sent request to create Client instance. Waiting for startup...')
            # assure the last created instance is running
            # waiting for full initialization is not necessary
            instance[-1].wait_until_running()
            logger.info('Successfully created Client instance')
        elif instance_type=='server':
            instance = self.resource.create_instances(
                MinCount=n_instances,
                MaxCount=n_instances,
                KeyName=self.__key_pair,
                UserData=startup_script,
                **ec2_config['server']
            )
            logger.info('Sent request to create Server instance. Waiting for initialization...')
            # assure the last created instance is running and fully initialized before moving to the next steps
            waiter = self.client.get_waiter('instance_status_ok')
            waiter.wait(
                Filters=[
                    {
                        'Name': 'instance-status.reachability',
                        'Values': ['passed']
                    },
                ],
                InstanceIds=[i.instance_id for i in instance]
            )
            logger.info('Successfully created Server instance')
        
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
        response = self.resource.terminate_instances(InstanceIds=instance_ids)
        return response
    

    def open_port(self, default_group: bool = True, port_number: int = 8080) -> dict:
        '''
        Add rule to the security group to open port

        Parameters
        ----------

        default_group : bool = True
            If the rule should be added to the default security group \n
            The newly started ec2 instance is attached to the default security group by default.
            Default value of this parameter is `True`, since the port 8080 should be opened prior to the instance start-up

        port_number : int
            Number of port to open \n
            8080 is the port demanded by Flower \n
            22 is the one required for SSH connections in case it is of interest

        Returns
        -------

        Operation response in a JSON format
        '''
        if default_group:
            # access details of default security group
            response = self.client.describe_security_groups(GroupNames=['default'])
            # get its ID
            security_group_id = response['SecurityGroups'][0]['GroupId']
        else:
            # create new security group
            sg_response = self.__create_security_group()
            # get its ID
            security_group_id = sg_response['GroupId']  
        try:   
            # compile the request to open port
            response = self.client.authorize_security_group_ingress(
                GroupId=security_group_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': port_number,
                        'ToPort': port_number,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}] # allows TCP traffic on port 8080 from any IP address (0.0.0.0/0)
                    }
                ]
            )
            return response
        except ClientError:
            # request failed
            return {'warning': 'rule already created, skipping action'}


    def __create_security_group(self, group_name: str = 'flower-ec2-group', description: str = 'group to open port 8080 on flower server instance') -> dict:
        '''
        Create the security group

        Parameters
        ----------

        group_name : str = 'flower-ec2-group'
            Group name
        
        description : str = 'group to open port 8080 on flower server instance'
            Group description

        Returns
        -------

        Operation response in a JSON format
        '''
        response = self.client.create_security_group(
            GroupName=group_name,
            Description=description
        )
        return response
    

    def create_s3_bucket(self, bucket_name: str = 'houseware-images-federated-learning-simulation-log') -> dict:
        '''
        Create s3 bucket to store logs from Flower Federated Learning simultations

        Parameters
        ----------

        bucket_name : str = 'houseware-images-federated-learning-simulation-log'
            Name of the bucket where logs will be stored
        
        Returns
        -------

        Operation response in a JSON format
        '''
        logger.info(f"Checking if the bucket '{bucket_name}' already exists...")
        try:
            self.client.head_bucket(Bucket=bucket_name)
            return {'warning': f"Bucket '{bucket_name}' already exists. Skipping action"}
        except ClientError as e:
            logger.info('Bucket does not exist. Creating...')
            if e.response['Error']['Code'] == '404':
                try:
                    response = self.client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={
                            'LocationConstraint': self.__region
                        }
                    )
                    return response
                except Exception as create_error:
                    return {'Error creating bucket': create_error}
            else:
                return {'Error checking bucket existence': e}
            
    
    def write_to_s3_bucket(self, log_file: str, object_key: str, bucket_name: str = 'houseware-images-federated-learning-simulation-log') -> dict:
        '''
        Write Flower Federated Learning simulation log file to the bucket

        Parameters
        ----------
        log_file : str
            Name of the file with log

        object_key : str
            Path to create for the log file inside the bucket \n
            Follow naming convention in such way that it is evident with which parameters the simulation was launched

        bucket_name : str = 'houseware-images-federated-learning-simulation-log'
            Name of the bucket where logs will be stored
        
        Returns
        -------

        Operation response in a JSON format
        '''
        try:
            # Put the object (file) in the S3 bucket
            with open(log_file, 'r') as f:
                log = f.read()
            log_binary = ' '.join(format(ch, 'b') for ch in bytearray(log))
            response = self.client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=log_binary
            )
            logger.info('Sent request to write the log to the bucket. Waiting until the log appears in bucket...')
            return response
        except Exception as e:
            return {'Error writing object to S3': e}