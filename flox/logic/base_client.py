from flox.common import ConfigFile, FitIns, NDArrays, XYData


class FloxClientLogic:
    def on_model_receive(self) -> None:
        """DocString"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def on_data_retrieve(self, config: ConfigFile) -> XYData:
        """DocString"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    # def on_data_prepare(self):
    #     """DocString"""
    #     raise NotImplementedError('Abstract class method. Cannot be called directly.')

    def on_model_fit(self, ins: FitIns) -> NDArrays:
        """DocString"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def on_model_send(self):
        """DocString"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def run_round(self):
        """DocString"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")
