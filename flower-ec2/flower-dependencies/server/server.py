import flwr as fl, sys, argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run Flower server')
    parser.add_argument('--public_ip', type=str, help='Public IP address of the Flower server ec2 instance', required=True)
    args = parser.parse_args()
    public_ip = args.public_ip

    fl.server.start_server(
        server_address=f'{public_ip}:8080',
        config=fl.server.ServerConfig(num_rounds=3)
    ) 