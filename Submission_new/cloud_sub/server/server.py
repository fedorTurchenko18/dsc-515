import flwr as fl

fl.server.start_server(
    server_address='[::]:8080',
    config=fl.server.ServerConfig(num_rounds=3)
)