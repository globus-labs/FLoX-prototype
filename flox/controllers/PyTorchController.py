from flox.controllers.MainController import MainController


class PyTorchController(MainController):
    def __init__(self, *args, testloader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.testloader = testloader

    def create_config(self, num_s, num_epoch, path_d):
        config = {
            "num_samples": num_s,
            "epochs": num_epoch,
            "path_dir": path_d,
            "dataset_name": self.dataset_name,
            "preprocess": self.preprocess,
        }

        return config

    def on_model_update(self, updated_weights) -> None:
        self.model_trainer.set_weights(updated_weights)

    def on_model_evaluate(self):
        results = self.model_trainer.evaluate(self.testloader)
        print(results)
        return results
