from flox.controllers.controller import MainController


class TensorflowController(MainController):
    def create_config(self, num_s, num_epoch, path_d):
        model_architecture = self.model_trainer.get_architecture(self.global_model)
        model_weights = self.model_trainer.get_weights(self.global_model)

        config = {
            "num_samples": num_s,
            "epochs": num_epoch,
            "path_dir": path_d,
            "data_source": self.data_source,
            "dataset_name": self.dataset_name,
            "preprocess": self.preprocess,
            "architecture": model_architecture,
            "weights": model_weights,
            "x_train_filename": self.x_train_filename,
            "y_train_filename": self.y_train_filename,
            "input_shape": self.input_shape,
        }

        return config

    def on_model_update(self, updated_weights) -> None:
        self.model_trainer.set_weights(self.global_model, updated_weights)

    def on_model_evaluate(self):
        if self.x_test is not None and self.y_test is not None:
            results = self.model_trainer.evaluate(
                self.global_model, self.x_test, self.y_test
            )
            print(results)
            return results
        else:
            print("Skipping evaluation, no x_test and/or y_test provided")
            return False
