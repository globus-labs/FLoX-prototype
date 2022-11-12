from abc import ABC


class FloxClientLogic(ABC):
    def on_model_receive():
        """DocString"""

    def on_data_retrieve():
        """DocString"""

    def on_data_prepare():
        """DocString"""

    def on_model_fit(trainer_obj):
        """DocString"""

    def on_model_send():
        """DocString"""

    def run_round():
        """DocString"""
