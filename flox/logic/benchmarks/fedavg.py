from numpy import array
from flox.logic.base_client import FloxClientLogic
from flox.logic.base_server import FloxServerLogic


class FedAvgServer(FloxServerLogic):
     
    def on_model_init():
        pass

    def on_model_broadcast():
        pass

    def on_model_aggr():
        pass


class FedAvgClient(FloxClientLogic):
    
    def on_model_recv():
        pass

    def on_data_fetch():
        pass

    def on_model_fit():
        pass

    def on_model_send():
        pass