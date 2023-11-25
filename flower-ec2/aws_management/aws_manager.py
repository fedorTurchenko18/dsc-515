import boto3, yaml, os, datetime, uuid
from botocore.exceptions import ClientError
from typing import Literal, List, Dict, Union
from loguru import logger

ec2_config_path = os.path.abspath(__file__)
ec2_config_path = ec2_config_path[:ec2_config_path.rindex('/')]
with open(f'{ec2_config_path}/ec2_config.yaml', 'rb') as f:
    ec2_config = yaml.safe_load(f)
BUCKET_NAME = str(uuid.uuid4())

class AWSManager:
    '''
    Class for managing AWS services
    '''
    def __init__(
            self,
            service: Literal['ec2', 's3', 'cloudwatch'],
            aws_access_key: str,
            aws_secret_access_key: str,
            aws_session_token: str,
            aws_region: Literal['us-east-1', 'us-west-2'],
            aws_key_pair: str
    ) -> None:
        '''
        Parameters
        ----------

        service : Literal['ec2', 's3', 'cloudwatch']
            Service to create client and resource for

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
            logger.info('Succesfully created Client ec2 instance')
        elif instance_type=='server':
            instance = self.resource.create_instances(
                MinCount=n_instances,
                MaxCount=n_instances,
                KeyName=self.__key_pair,
                UserData=startup_script,
                **ec2_config['server']
            )
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
            logger.info('Succesfully created Server ec2 instance')
        
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
    

    def create_s3_bucket(self, bucket_name: str = BUCKET_NAME) -> dict:
        '''
        Create s3 bucket to store logs from Flower Federated Learning simultations

        Parameters
        ----------

        bucket_name : str = `uuid.uuid4()`
            Name of the bucket where logs will be stored. UUID is used to assure unique bucket name
        
        Returns
        -------

        Operation response in a JSON format
        '''
        try:
            self.client.head_bucket(Bucket=bucket_name)
            return {'warning': f"Bucket '{bucket_name}' already exists. Skipping action"}
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                try:
                    response = self.client.create_bucket(
                        Bucket=bucket_name
                    )
                    return response
                except Exception as create_error:
                    return {'Error creating bucket': create_error}
            else:
                return {'Error checking bucket existence': e}
            
    
    def write_to_s3_bucket(self, log_file: str, object_key: str, bucket_name: str = BUCKET_NAME) -> dict:
        '''
        Write Flower Federated Learning simulation log file to the bucket

        Parameters
        ----------
        log_file : str
            Name of the file with log

        object_key : str
            Path to create for the log file inside the bucket \n
            Follow naming convention in such way that it is evident with which parameters the simulation was launched

        bucket_name : str = `uuid.uuid4()`
            Name of the bucket where logs will be stored. UUID is used to assure unique bucket name
        
        Returns
        -------

        Operation response in a JSON format
        '''
        try:
            # Put the object (file) in the S3 bucket
            with open(log_file, 'r') as f:
                log = f.read()
            log_binary = str.encode(log)
            response = self.client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=log_binary
            )
            return response
        except Exception as e:
            return {'Error writing object to S3': e}
        
    
    def download_from_s3_bucket(self, local_file_path: str, object_key: str, bucket_name: str = BUCKET_NAME) -> dict:
        '''
        Download Flower Federated Learning simulation log file from the bucket

        Parameters
        ----------
        local_file_path : str
            Local path to save the log to

        object_key : str
            Path to create for the log file inside the bucket \n
            Follow naming convention in such way that it is evident with which parameters the simulation was launched

        bucket_name : str = `uuid.uuid4()`
            Name of the bucket where log is stored. UUID is used to assure unique bucket name
        
        Returns
        -------

        Operation response in a JSON format
        '''
        try:
            response = self.resource.meta.client.download_file(bucket_name, object_key, local_file_path)
        except ClientError as e:
            return {'error': e}
        return response
    

    def delete_s3_bucket(self, bucket_name: str = BUCKET_NAME) -> dict:
        '''
        Delete the bucket with Flower Federated Learning simulation log file

        Parameters
        ----------

        bucket_name : str = `uuid.uuid4()`
            Name of the bucket where log is stored. UUID is used to assure unique bucket name
        
        Returns
        -------

        Operation response in a JSON format
        '''
        bucket_contents = self.client.list_objects(Bucket=bucket_name)
        objects_to_delete = [{k: v} for d in bucket_contents['Contents'] for k, v in d.items() if k=='Key']
        objects_delete_request = self.client.delete_objects(
            Bucket=bucket_name,
            Delete={
                'Objects': objects_to_delete,
                'Quiet': False
            }
        )
        bucket_delete_response = self.client.delete_bucket(Bucket=bucket_name)
        return bucket_delete_response
        
    
    def get_metric_statistic_cloudwatch(
            self, instance_id: str, start_time: datetime.datetime, end_time: datetime.datetime,
            metric_name: str = 'CPUUtilization', namespace: str = 'AWS/EC2', period: int = 10, stats: List[str]=['Average']
    ) -> Dict[
        Literal['timestamp', 'cpu_utilization', 'error'],
        Union[Literal['cpu utilization estimation failed'], datetime.datetime, float]
    ]:
        '''
        WARNING: Deprecated Method
        --------------------------

        Cloudwatch delay does not allow for real-time metrics tracking. 
        Deprecated in favor of `psutil`

        ---

        Measure ec2 instance metric \n
        Default usage is measuring CPU usage at Flower Client during `fit()` and `evaluate()` operations. 
        It is calculated by averaging 10-seconds long windows by default

        Parameters
        ----------

        instance_id : str
            ID of ec2 instance to measure on
        
        start_time : datime.datetime
            Starting point to measure from

        end_time : datetime.datetime
            When to finish measuring

        metric_name : str
            The name of the metric, with or without spaces

        namespace : str
            The namespace of the metric, with or without spaces

        period : int
            The granularity, in seconds, of the returned data points

        statistic : List[str]
            The metric statistics

        Returns
        -------

        Dict of either CPU usage along with timestamp or error message
        '''
        response = self.client.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=[
                {
                    'Name': 'InstanceId',
                    'Value': instance_id
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=period,
            Statistics=stats
        )
        try:
            cpu_usage = {'timestamp': [], 'cpu_utilization': []}
            for datapoint in response['Datapoints']:
                cpu_usage['timestamp'].append(datapoint['Timestamp'])
                cpu_usage['cpu_utilization'].append(datapoint['Average'])
            return cpu_usage
        except KeyError:
            return {'error': 'cpu utilization estimation failed'}