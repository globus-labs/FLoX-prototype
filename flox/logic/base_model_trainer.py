from abc import ABC, abstractmethod


class BaseModelTrainer(ABC):
    @abstractmethod
    def fit(self):
        """DocString"""

    def evaluate(self):
        """DocString"""

    def get_weights(self):
        """DocString"""

    def set_weights(self):
        """DocString"""

    def create_model(self):
        """DocString"""

    def compile_model(self):
        """DocString"""

    def get_architecture(self):
        """DocString"""
