# conda activate py37
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from funcx.sdk.client import FuncXClient
from funcx.sdk.executor import FuncXExecutor

def federated_decorator(func):
    def wrapper(*args, **kwargs):
        fx = FuncXExecutor(FuncXClient())
        tasks = []
        for e in kwargs["endpoint_ids"]:
            tasks.append(fx.submit(func, 
                                   **kwargs,
                                    endpoint_id=e))
        return tasks
    
    return wrapper

# path_dir='/home/pi/datasets', x_train_path="mnist_x_train.npy", y_train_path="mnist_y_train.npy"
def get_local_data(x_train_path, y_train_path, path_dir=".", preprocess=None, preprocessing_function=None):
    '''
    Returns (x_train, y_train) given the edge directory and filenames.
    
    '''
    import numpy as np
    import os
    import collections
    
    # construct the path
    x_train_path_file = os.sep.join([path_dir, x_train_path])
    y_train_path_file = os.sep.join([path_dir, y_train_path])

    # load the files
    with open(x_train_path_file, 'rb') as f:
        x_train = np.load(f)
        
    with open(y_train_path_file, 'rb') as f:
        y_train = np.load(f)

    # if preprocess is True & the function is valid, preprocess the data
    if preprocess:
        # check if a valid function was given
        if not preprocessing_function or not callable(preprocessing_function):
            raise TypeError('preprocessing_function is not a function. Please provide a valid function in your call')
        
        (x_train, y_train) = preprocessing_function(x_train, y_train)

    return (x_train, y_train)


def get_keras_data(keras_dataset, preprocess=True, num_samples=None):
    '''
    Returns (x_train, y_train) of a chosen built-in Keras dataset. Also preprocesses the image data by default.
    Options: ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'reuters', 'boston_housing']
    
    '''
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
        (x_train, y_train), _ = dataset_mapping[keras_dataset].load_data()

        # take a random set of images
        if num_samples:
            idx = np.random.choice(np.arange(len(x_train)), num_samples, replace=True)
            x_train = x_train[idx]
            y_train = y_train[idx]

        # do default image processing for built-in Keras images
        if preprocess:
            if keras_dataset in image_datasets:
                # Scale images to the [0, 1] range
                x_train = x_train.astype("float32") / 255

                # Make sure images have shape (num_samples, x, y, 1) if working with MNIST images
                if x_train.shape[-1] not in [1, 3]:
                    x_train = np.expand_dims(x_train, -1)

                # convert class vectors to binary class matrices
                if keras_dataset == 'cifar100':
                    num_classes=100
                else:
                    num_classes=10
                    
                y_train = keras.utils.to_categorical(y_train, num_classes)

        return (x_train, y_train)

def train_default_model(json_model_config, 
                global_model_weights,
                x_train,
                y_train,
                epochs=10,
                loss="categorical_crossentropy",
                optimizer="adam", 
                metrics=["accuracy"]):

    # import dependencies
    from tensorflow import keras
    import numpy as np

    # create the model
    model = keras.models.model_from_json(json_model_config)

    # compile the model and set weights to the global model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.set_weights(global_model_weights)

    # train the model on the local data and extract the weights
    model.fit(x_train, y_train, epochs=epochs)
    model_weights = model.get_weights()

    # transform to a numpy array
    np_model_weights = np.asarray(model_weights, dtype=object)

    return np_model_weights


def create_training_function(train_model=train_default_model, 
                            data_source: str = "keras",
                            preprocessing_function=None,
                            path_dir='/home/pi/datasets', 
                            x_train_path="mnist_x_train.npy", 
                            y_train_path="mnist_y_train.npy", 
                            preprocess_local=True, 
                            keras_dataset = "mnist", 
                            preprocess_keras=True, 
                            loss="categorical_crossentropy",
                            optimizer="adam", 
                            metrics=["accuracy"],
                            get_keras_data=get_keras_data,
                            get_local_data=get_local_data,
                            get_custom_data=None,
                            **kwargs
):
    
    def training_function(json_model_config, 
                          global_model_weights, 
                          num_samples=None,
                          epochs=10,
                          **kwargs
):

        # import all the dependencies required for funcX functions)
        import numpy as np

        if data_source == 'local':
            (x_train, y_train) = get_local_data(path_dir=path_dir, 
                          x_train_path=x_train_path, 
                          y_train_path=y_train_path, 
                          preprocess=preprocess_local, 
                          preprocessing_function=preprocessing_function)

        elif data_source == 'keras':
            (x_train, y_train) = get_keras_data(keras_dataset, 
                                                preprocess_keras, 
                                                num_samples)

        elif data_source == 'custom':
            if callable(get_custom_data):
                (x_train, y_train) = get_custom_data()
            else:
                raise TypeError('preprocessing_function is not a function. Please provide a valid function in your call')

        else:
            raise Exception("Please choose one of data sources: ['local', 'keras', 'pass']")

        # consider switching to args* and kwargs** in case of a custom training function

        model_weights = train_model(json_model_config=json_model_config, 
                                            global_model_weights=global_model_weights, 
                                            x_train=x_train, 
                                            y_train=y_train,
                                            epochs=epochs,
                                            loss=loss,
                                            optimizer=optimizer, 
                                            metrics=metrics)

        return {"model_weights":model_weights, "samples_count": x_train.shape[0]}
    
    return training_function

def get_edge_weights(sample_counts):
    '''
    Returns weights for each model to find the weighted average 
    '''
    total = sum(sample_counts)
    fractions = sample_counts/total
    return fractions

def federated_average(global_model, 
                      endpoint_ids, 
                      num_samples=100,
                      epochs=10,
                      weighted=False,
                      train_model=train_default_model, 
                      data_source: str = "keras",
                      preprocessing_function=None,
                      path_dir='/home/pi/datasets', 
                      x_train_path="mnist_x_train.npy", 
                      y_train_path="mnist_y_train.npy", 
                      preprocess_local=None, 
                      keras_dataset = "mnist", 
                      preprocess_keras=True, 
                      loss="categorical_crossentropy",
                      optimizer="adam", 
                      metrics=["accuracy"],
                      get_custom_data=None,
                      **kwargs):

    fx = FuncXExecutor(FuncXClient())

    # get the model's architecture and weights
    json_config = global_model.to_json()
    gm_weights = global_model.get_weights()
    gm_weights_np = np.asarray(gm_weights, dtype=object)

    # compile the training function
    training_function = create_training_function(train_model=train_model, 
                                                data_source = data_source,
                                                path_dir=path_dir, 
                                                x_train_path=x_train_path, 
                                                y_train_path=y_train_path, 
                                                preprocess_local=preprocess_local, 
                                                preprocessing_function=preprocessing_function,
                                                keras_dataset = keras_dataset, 
                                                preprocess_keras=preprocess_keras, 
                                                loss=loss,
                                                optimizer=optimizer, 
                                                metrics=metrics,
                                                get_custom_data=get_custom_data)
    
    # train the MNIST model on each of the endpoints and return the result, sending the global weights to each edge
    tasks = []
    for e in endpoint_ids:
        tasks.append(fx.submit(training_function, 
                                json_model_config=json_config, 
                                global_model_weights=gm_weights_np, 
                                num_samples=num_samples,
                                epochs=epochs,
                                endpoint_id=e))
    
    # extract weights from each edge model
    model_weights = [t.result()["model_weights"] for t in tasks]
    
    if weighted:
        # get the weights
        sample_counts = np.array([t.result()["samples_count"] for t in tasks])
        edge_weights = get_edge_weights(sample_counts)
        
        print(f"Model Weights: {edge_weights}")
        # find weighted average
        average_weights = np.average(model_weights, weights=edge_weights, axis=0)
        
    else:
        # simple average of the weights
        average_weights = np.mean(model_weights, axis=0)
    
    # assign the weights to the global_model
    global_model.set_weights(average_weights)

    print('Trained Federated Model')

    return global_model


def default_model_inference(json_model_config, 
                global_model_weights,
                x_train,
                y_train,
                loss="categorical_crossentropy",
                optimizer="adam", 
                metrics=["accuracy"],
                **extra_compiler_arguments):
    # import dependencies
    from tensorflow import keras
    import numpy as np

    # create the model
    model = keras.models.model_from_json(json_model_config)

    # compile the model and set weights to the global model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics, **extra_compiler_arguments)
    model.set_weights(global_model_weights)

    # train the model on the local data and extract the weights
    predictions = model.predict(x_train)

    # save it in a file

    return "Thanks"


def create_inference_function(data_source: str = "keras",
                            preprocessing_function=None,
                            path_dir='/home/pi/datasets', 
                            x_train_path="mnist_x_train.npy", 
                            y_train_path="mnist_y_train.npy", 
                            preprocess_local=True, 
                            keras_dataset = "mnist", 
                            preprocess_keras=True, 
                            loss="categorical_crossentropy",
                            optimizer="adam", 
                            metrics=["accuracy"],
                            get_keras_data=get_keras_data,
                            get_local_data=get_local_data,
                            get_custom_data=None,
                            **kwargs
):
    
    def inference_function(json_model_config, 
                          global_model_weights, 
                          num_samples=None,
                          loops=2,
                          time_interval=0,
                          **kwargs
):

        # import dependencies
        from tensorflow import keras
        import numpy as np
        import time

        # create the model
        model = keras.models.model_from_json(json_model_config)

        # compile the model and set weights to the global model
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.set_weights(global_model_weights)

        for i in range(loops):
        # train the model on the local data and extract the weights

            if data_source == 'local':
                (x_train, y_train) = get_local_data(path_dir=path_dir, 
                            x_train_path=x_train_path, 
                            y_train_path=y_train_path, 
                            preprocess=preprocess_local, 
                            preprocessing_function=preprocessing_function)

            elif data_source == 'keras':
                (x_train, y_train) = get_keras_data(keras_dataset, 
                                                    preprocess_keras, 
                                                    num_samples)

            elif data_source == 'custom':
                if callable(get_custom_data):
                    (x_train, y_train) = get_custom_data()
                else:
                    raise TypeError('preprocessing_function is not a function. Please provide a valid function in your call')

            else:
                raise Exception("Please choose one of data sources: ['local', 'keras', 'pass']")

            predictions = model.predict(x_train)

            # wait for time_interval seconds 
            time.sleep(time_interval)

            # save the stats in a file for future retrieval

        return "The inference process is over"
    
    return inference_function