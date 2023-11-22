if __name__=='__main__':

    import os
    from dotenv import load_dotenv
    from ec2_management.ec2_manager import EC2Manager

    load_dotenv()
    AWS_ACCESS_KEY = os.environ['AWS_LAB_ACCESS_KEY']
    AWS_SECRET_ACCESS_KEY = os.environ['AWS_LAB_SECRET_ACCESS_KEY']
    AWS_SESSION_TOKEN = os.environ['AWS_LAB_SESSION_TOKEN']
    AWS_REGION = os.environ['AWS_REGION']
    AWS_KEY_PAIR = os.environ['AWS_KEY_PAIR']

    manager = EC2Manager(
        aws_access_key=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        aws_region=AWS_REGION,
        aws_key_pair=AWS_KEY_PAIR
    )

    # add rule to the default security group to open port 8080 and 22
    manager.open_port(default_group=True, port_number=8080)
    manager.open_port(default_group=True, port_number=22)

    # create and run the server ec2 instance
    with open('flower-dependencies/server/server_startup.txt', 'r') as f:
        server_startup_script = f.read()
    flower_server = manager.create_instance('server', startup_script=server_startup_script)

    # Extract the public IP address from the response
    response = manager.client.describe_instances(InstanceIds=[flower_server[0].instance_id])
    server_public_ip_address = response['Reservations'][0]['Instances'][0]['PublicIpAddress']

    # read client start-up script to modify the server public IP there
    with open('flower-dependencies/client/client_startup.txt', 'r') as f:
        client_startup_script = [line for line in f.readlines()]
    # modify the line
    client_startup_script = [
        line.replace(line[line.find('=')+len('='):line.find('\\')], server_public_ip_address) \
            if 'export PUBLIC_IP' in line else line \
                for line in client_startup_script
    ]
    # write new version of the script
    with open('flower-dependencies/client/client_startup.txt', 'w') as f:
        f.writelines(client_startup_script)

    # create and run the clients ec2 instances
    with open('flower-dependencies/client/client_startup.txt', 'r') as f:
        client_startup_script = f.read()
    flower_clients = manager.create_instance('clients', startup_script=client_startup_script, n_instances=2)