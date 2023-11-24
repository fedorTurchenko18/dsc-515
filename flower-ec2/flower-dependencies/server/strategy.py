import flwr as fl
from typing import Literal, Callable


class ServerStrategy:
    def __init__(
            self,
            fl_strategy: Literal[
                'FedAvg',
                'FedAvgM',
                'FedAdaGrad',
                'FedAdam'
            ],
            min_available_clients: int,
            min_fit_clients: int,
            min_evaluate_clients: int,
            evaluate_fn: Callable,
            initial_parameters: fl.common.Parameters,
            **kwargs
    ) -> None:
        '''
        Parameters of `fl.server.strategy.Strategy` and custom ones

        Such `fl.server.strategy.Strategy` parameters as `fraction_fit` and `fraction_evaluate` which account for
        share of clients are not used, since the simulation implies working on
        small number of clients compared to real workflows

        Parameters
        ----------

        fl_strategy : Literal['FedAvg', 'FedAvgM', 'FedAdaGrad', 'FedAdam']
            One of such flower server strategies as `FedAvg`, `FedAvgM`, `FedAdaGrad`, `FedAdam` \n
            Other ones are possible but out of the scope of simulation interest

        min_available_clients : int
            Minimum number of clients used during training

        min_fit_clients : int
            Minimum number of clients used during validation

        min_evaluate_clients : int
            Minimum number of total clients in the system

        evaluate_fn : Callable
            Function used for validation \n
            Docs: \n
            Centralized Evaluation (or server-side evaluation) is conceptually simple: it works the same way that evaluation in centralized machine learning does. If there is a server-side dataset that can be used for evaluation purposes, then that’s great. We can evaluate the newly aggregated model after each round of training without having to send the model to clients. We’re also fortunate in the sense that our entire evaluation dataset is available at all times.\n
            Makes sense in simulation as small number of clients is always available

        initial_parameters : fl.common.Parameters
            Initial global model parameters \n
            Docs: \n
            Flower, by default, initializes the global model by asking one random client for the initial parameters. In many cases, we want more control over parameter initialization though.
        '''
        self.strategy = fl_strategy(
            min_available_clients = min_available_clients,
            min_fit_clients = min_fit_clients,
            min_evaluate_clients = min_evaluate_clients,
            evaluate_fn = evaluate_fn,
            initial_parameters = initial_parameters,
            **kwargs
        )