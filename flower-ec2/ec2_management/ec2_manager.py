import boto3, yaml
from botocore.exceptions import ClientError
from typing import Literal, List

with open('ec2_management/ec2_config.yaml', 'rb') as f:
    ec2_config = yaml.safe_load(f)


class EC2Manager:
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
        self.resource = boto3.resource(
            'ec2',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=aws_region
        )
        self.client = boto3.client(
            'ec2',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=aws_region
        )


    def create_instance(self, instance_type: Literal['server', 'clients'], startup_script: str, n_instances: int = 2):
        '''
        Create ec2 instance

        Parameters
        ----------

        instance_type : Literal['server', 'clients']
            Type of instance, respective to Flower's resource-server paradigm

        startup_script : str
            Script to run on ec2 instance start-up. 
            Should be read through context manager from .txt file
        
        n_instances : int = 2
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
        elif instance_type=='server':
            instance = self.resource.create_instances(
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


class EC2Instance(EC2Manager):
    '''
    Class for accessing and performing actions on indivdual ec2 instance
    '''
    def __init__(self, instance_id: str, aws_access_key: str, aws_secret_access_key: str, aws_region: Literal['us-east-1', 'us-west-2'], key_pair: str) -> None:
        '''
        instance_id : str
            ID of the instance to manage
        '''
        super().__init__(aws_access_key, aws_secret_access_key, aws_region, key_pair)
        self.instance_id = instance_id

    
    # def assign_security_group(self) -> None:
    #     '''
    #     Create and assign security group with opened 8080 port rule to instance
    #     '''
    #     # Get the current security group IDs associated with the instance
    #     response = self.resource.describe_instances(InstanceIds=[self.instance_id])
    #     current_security_group_ids = response['Reservations'][0]['Instances'][0]['SecurityGroups']

    #     # Extract the existing security group IDs
    #     existing_security_group_ids = [sg['GroupId'] for sg in current_security_group_ids]
        
    #     # create security group
    #     sg_response = self.__create_security_group()
    #     # get its ID
    #     security_group_id_to_add = sg_response['GroupId']
    #     # open the 8080 port
    #     _ = self.__open_port(security_group_id=security_group_id_to_add)

    #     # Add the new security group ID to the list if it's not already present
    #     if security_group_id_to_add not in existing_security_group_ids:
    #         existing_security_group_ids.append(security_group_id_to_add)

    #     # Modify the instance to include the updated security group IDs
    #     self.resource.modify_instance_attribute(
    #         InstanceId=self.instance_id,
    #         Groups=existing_security_group_ids
    #     )