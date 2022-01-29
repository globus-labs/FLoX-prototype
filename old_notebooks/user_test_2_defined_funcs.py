# works well when tested on Linux-based system. 
# Causes the 'package fcntl not found' error on windows
from flx_module.funcx_federated import hello_world
from flx_module.funcx_federated import federated_average
import flx_module.funcx_federated

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

endpoint_ids = ['00929e1a-ccc5-40be-8b04-c171f132f7b2', '11983ca1-2d45-40d1-b5a2-8736b3544dea']
batch_size = 128
epochs = 5
input_shape = (28, 28, 1)
num_classes = 10

global_model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
    )

global_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

def get_new_data():
    from tensorflow import keras
    import numpy as np

    num_samples = 20

    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    
    # take a random set of images
    idx = np.random.choice(np.arange(len(x_train)), num_samples, replace=True)
    x_train = x_train[idx]
    y_train = y_train[idx]

    return (x_train, y_train)

federated_average(global_model=global_model, 
                  endpoint_ids=endpoint_ids,
                  get_data=get_new_data,
                  weighted=False)
