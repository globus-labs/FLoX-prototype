from abc import ABC


class FloxClientLogic(ABC):

    def on_model_recv():
        pass

    def on_data_fetch():
        pass

    def on_model_fit(trainer_obj):
        pass

    def on_model_send():
        pass