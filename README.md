# FLoX

FLoX (**F**ederated **L**earning on func**X**) is a Python library for serverless Federated Learning experiments.

This is initial documentation that will be soon expanded. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install flox.

```bash
# seems like flox is taken, will update the name soon
pip install flox
```

## Usage
For a full example, see this [Google Colab tutorial](https://colab.research.google.com/drive/19X1N8E5adUrmeE10Srs1hSQqCCecv23m?usp=sharing).

```python
from flox.flox import federated_learning

# performs 5 rounds of Federated Learning train global_model on given endpoint_ids
# uses 10 epochs and 100 samples from fashion_mnist dataset for training
federated_learning(global_model=global_model, 
                  endpoint_ids=endpoint_ids,
                  loops=5,
                  epochs=10,
                  keras_dataset="fashion_mnist", 
                  num_samples=100, 
                  input_shape=(32, 28, 28, 1))
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)