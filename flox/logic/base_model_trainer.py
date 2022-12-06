class BaseModelTrainer:
    def fit(self):
        """DocString"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def evaluate(self):
        """DocString"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def get_weights(self):
        """DocString"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def set_weights(self):
        """DocString"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    # def create_model(self):
    #     """DocString"""
    #     raise NotImplementedError('Abstract class method. Cannot be called directly.')

    # def compile_model(self):
    #     """DocString"""
    #     raise NotImplementedError('Abstract class method. Cannot be called directly.')

    # def get_architecture(self):
    #     """DocString"""
    #     raise NotImplementedError('Abstract class method. Cannot be called directly.')
