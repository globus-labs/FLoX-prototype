from funcx_federated import create_inference_function, federated_decorator, create_training_function

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def get_test_data(keras_dataset='mnist', num_samples=None, preprocess=True, preprocessing_function=None, **kwargs):
    """
    Returns (x_test, y_test) of a chosen built-in Keras dataset. 
    Also preprocesses the image datasets (mnist, fashion_mnist, cifar10, cifar100) by default.

    Parameters
    ----------
    keras_dataset: str
        one of the available Keras datasets: 
        ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'reuters', 'boston_housing']

    num_samples: int 
        randomly samples n data points from (x_test, y_test). Set to None by default.

    preprocess: boolean
        if True, preprocesses (x_test, y_test) 

    preprocessing_function: function
        a custom user-provided function that processes (x_test, y_test) and outputs 
        a tuple (x_test, y_test)

    Returns
    -------

    """
    from tensorflow import keras
    import numpy as np

    available_datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'reuters', 'boston_housing']
    dataset_mapping= {
        'mnist': keras.datasets.mnist,
        'fashion_mnist': keras.datasets.fashion_mnist,
        'cifar10': keras.datasets.cifar10,
        'cifar100': keras.datasets.cifar100,
        'imdb': keras.datasets.imdb,
        'reuters': keras.datasets.reuters,
        'boston_housing': keras.datasets.boston_housing
    }
    image_datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']

    # check if the dataset exists
    if keras_dataset not in available_datasets:
        raise Exception(f"Please select one of the built-in Keras datasets: {available_datasets}")

    else:
        _, (x_test, y_test) = dataset_mapping[keras_dataset].load_data()

        # take a random set of images
        if num_samples:
            idx = np.random.choice(np.arange(len(x_test)), num_samples, replace=True)
            x_test = x_test[idx]
            y_test = y_test[idx]

        if preprocess:
            if preprocessing_function and callable(preprocessing_function):
                (x_test, y_test) = preprocessing_function(x_test, y_test)

            else:
                # do default image processing for built-in Keras images    
                if keras_dataset in image_datasets:
                    # Scale images to the [0, 1] range
                    x_test = x_test.astype("float32") / 255

                    # Make sure images have shape (num_samples, x, y, 1) if working with MNIST images
                    if x_test.shape[-1] not in [1, 3]:
                        x_test = np.expand_dims(x_test, -1)

                    # convert class vectors to binary class matrices
                    if keras_dataset == 'cifar100':
                        num_classes=100
                    else:
                        num_classes=10
                        
                    y_test = keras.utils.to_categorical(y_test, num_classes)

        return (x_test, y_test)


def get_train_data(dataset='mnist', num_samples=1000):
    dataset_mapping= {
        'mnist': keras.datasets.mnist,
        'fashion_mnist': keras.datasets.fashion_mnist,
        'cifar10': keras.datasets.cifar10,
        'cifar100': keras.datasets.cifar100,
        'imdb': keras.datasets.imdb,
        'reuters': keras.datasets.reuters,
        'boston_housing': keras.datasets.boston_housing
    }

    num_classes = 10
    (x_test, y_test), _= dataset_mapping[dataset].load_data()
    
    idx = np.random.choice(np.arange(len(x_test)), num_samples, replace=True)
    x_test = x_test[idx]
    y_test = y_test[idx]
    
    x_test = x_test.astype("float32") / 255
    x_test = np.expand_dims(x_test, -1)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_test, y_test)

def eval_model(m, x, y):
    ''' evaluate model on dataset x,y'''
    score = m.evaluate(x, y, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

def preprocess_data(x_test, y_test, num_samples=100):
    from tensorflow import keras
    import numpy as np

    num_classes = 10

    # take a random set of images
    idx = np.random.choice(np.arange(len(x_test)), num_samples, replace=True)
    x_test = x_test[idx]
    y_test = y_test[idx]

    # Scale images to the [0, 1] range
    x_test = x_test.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_test = np.expand_dims(x_test, -1)
    print("x_test shape:", x_test.shape)
    print(x_test.shape[0], "train samples")

    # convert class vectors to binary class matrices
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_test, y_test)

x_test, y_test = get_test_data()
endpoint_ids = ['11983ca1-2d45-40d1-b5a2-8736b3544dea', '11983ca1-2d45-40d1-b5a2-8736b3544dea']
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

json_config = global_model.to_json()
model_weights = np.asarray(global_model.get_weights(), dtype=object)


# deployment
federated_inference = federated_decorator(create_inference_function())
federated_inference(json_model_config=json_config, 
                          global_model_weights=model_weights, 
                          num_samples=10,
                          loops=2,
                          endpoint_ids=endpoint_ids)

# training one time without the fed. average and without finding the average
federated_learning = federated_decorator(create_training_function())
result = federated_learning(json_model_config=json_config, 
                          global_model_weights=model_weights, 
                          num_samples=10,
                          endpoint_ids=endpoint_ids)


# training in a loop without the federated_average() function
endpoint_ids = ["6d2cc03e-565d-494b-9bdf-0ba0acdc606f", "11983ca1-2d45-40d1-b5a2-8736b3544dea"]
federated_learning = federated_decorator(create_training_function())

for i in range(5):
    gm_weights_np = np.asarray(global_model.get_weights(), dtype=object)
    tasks = federated_learning(json_model_config=json_config, 
                              global_model_weights=gm_weights_np, 
                              num_samples=10,
                              endpoint_ids=endpoint_ids)
    
    model_weights = [t.result()["model_weights"] for t in tasks]
    average_weights = np.mean(model_weights, axis=0)
    global_model.set_weights(average_weights)
    eval_model(global_model, x_test, y_test)