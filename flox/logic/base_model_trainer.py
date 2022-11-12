from abc import ABC


class BaseModelTrainer(ABC):

    arch: any

    def __init__(self, arch):
        self.arch = arch

    def fit(self):
        """DocString"""

    def evaluate(self):
        """DocString"""

    def get_weights(self):
        """DocString"""

    def set_weights(self):
        """DocString"""

    def build_model(self):
        """DocString"""

    def compile_model(self):
        """DocString"""

    def get_architecture(self):
        """DocString"""
