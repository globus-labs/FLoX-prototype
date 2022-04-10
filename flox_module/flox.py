import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time

from funcx.sdk.client import FuncXClient
from funcx.sdk.executor import FuncXExecutor


def get_edge_weights(sample_counts):
    """
    Takes an array of numbers and returns their fractions of the total number of samples
    Can be used to find weights for the weighted_average 

    Parameters
    ----------
    sample_counts: numpy array of integers

    Returns
    -------
    fractions: numpy array
    
    """
    total = sum(sample_counts)
    fractions = sample_counts/total
    return fractions

def eval_model(m, x, y, silent=False):
    """
    Evaluate the model on a datset

    Parameters
    ----------
    m: Tensorflow model

    x: numpy array
        dataset entries, e.g. x_test
    
    y: numpy array
        labels for the entries, e.g., y_test
    
    """
    score = m.evaluate(x, y, verbose=0)
    if not silent:
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
    return score[0], score[1]
def training_function(json_model_config, 
                      global_model_weights, 
                      num_samples=None,
                      epochs=10,     
                      data_source="keras",
                      preprocess=True,
                      keras_dataset="mnist",
                      input_shape=(32, 28, 28, 1),
                      loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"],                            
                      path_dir='/home/pi/datasets', 
                      x_train_name="mnist_x_train.npy", 
                      y_train_name="mnist_y_train.npy"
):
    """

    Parameters
    ----------
    json_model_config: str
        configuration of the TF model retrieved using model.to_json()

    global_model_weights: numpy array
        a numpy array with weights of the TF model

    num_samples: int
        if data_source="keras", randomly samples n data points from (x_train, y_train)

    epochs: int
        the number of epochs to train the model for

    Returns
    -------
    model_weights: numpy array
        updated TF model weights after training on the local data

    samples_counts: int
        the number of samples the model was trained on. 
        Can be used to find the weighted average by # of samples seen

    """
    # import all the dependencies required for funcX functions)
    from tensorflow import keras
    import numpy as np
    import os

    # retrieve (and optionally process) the data
    if data_source == 'keras':
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

            if preprocess:
                # do default image processing for built-in Keras images    
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
    elif data_source == 'local':        
        # construct the path
        x_train_path_file = os.sep.join([path_dir, x_train_name])
        y_train_path_file = os.sep.join([path_dir, y_train_name])

        # load the files
        with open(x_train_path_file, 'rb') as f:
            x_train = np.load(f)
            
        with open(y_train_path_file, 'rb') as f:
            y_train = np.load(f)

        # if preprocess is True & the function is valid, preprocess the data
        if preprocess:
            # check if a valid function was given
            depth = input_shape[3]
            image_size_y = input_shape[2]
            image_size_x = input_shape[1]

            if num_samples:
                idx = np.random.choice(np.arange(len(x_train)), num_samples, replace=True)
                x_train = x_train[idx]
                y_train = y_train[idx]
            
            x_train = x_train.reshape(len(x_train), image_size_x, image_size_y, depth)
            x_train = x_train / 255.0

    else:
        raise Exception("Please choose one of data sources: ['local', 'keras', 'custom']")

    # train the model
    # create the model
    model = keras.models.model_from_json(json_model_config)

    # compile the model and set weights to the global model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    #global_model_weights = np.asarray(global_model_weights, dtype=object)
    # this is a temporary fix for a bug on the testing side
    # where it says I need to build the model first   
    try:
        model.set_weights(global_model_weights)
    except:
        model.build(input_shape=input_shape)
        model.set_weights(global_model_weights)

    # train the model on the local data and extract the weights
    model.fit(x_train, y_train, epochs=epochs)
    model_weights = model.get_weights()

    # transform to a numpy array
    np_model_weights = np.asarray(model_weights, dtype=object)

    # return the updated weights and number of samples the model was trained on
    return {"model_weights":np_model_weights, "samples_count": x_train.shape[0]}

def federated_learning(global_model, 
                      endpoint_ids, 
                      num_samples=100,
                      epochs=5,
                      loops=1,
                      time_interval=0,
                      federated_mode="weighted_average",
                      data_source: str = "keras",
                      preprocess=False,
                      keras_dataset = "mnist",  
                      input_shape=(32, 28, 28, 1),
                      loss="categorical_crossentropy",
                      optimizer="adam", 
                      metrics=["accuracy"],
                      evaluation_function=eval_model,
                      x_test=None,
                      y_test=None,
                      path_dir='/home/pi/datasets', 
                      x_train_name="mnist_x_train.npy", 
                      y_train_name="mnist_y_train.npy"):
    """

    Facilitates federated learning on given endpoints

    Parameters
    ----------


    Returns
    -------

    Examples
    --------
    
    """
    fx = FuncXExecutor(FuncXClient())

    # compile the training function
    if type(num_samples) == int:
        num_samples = [num_samples]*len(endpoint_ids)

    if type(epochs) == int:
        epochs = [epochs]*len(endpoint_ids)
    
    for i in range(loops):
        # get the model's architecture and weights
        json_config = global_model.to_json()
        gm_weights = global_model.get_weights()
        gm_weights_np = np.asarray(gm_weights, dtype=object)

        # train the MNIST model on each of the endpoints and return the result, sending the global weights to each edge
        fx = FuncXExecutor(FuncXClient())
        tasks = []

        # for each endpoint, submit the function with **kwargs to it
        for e, num_s, num_epoch, path_d in zip(endpoint_ids, num_samples, epochs, path_dir): 
            tasks.append(fx.submit(training_function, 
                                   json_model_config=json_config, 
                                    global_model_weights=gm_weights_np, 
                                    num_samples=num_s,
                                    epochs=num_epoch,
                                    data_source=data_source,
                                    preprocess=preprocess,
                                    keras_dataset=keras_dataset,
                                    input_shape=input_shape,
                                    loss=loss,
                                    optimizer=optimizer,
                                    metrics=metrics,
                                    path_dir=path_d,
                                    x_train_name=x_train_name,
                                    y_train_name=y_train_name,
                                    endpoint_id=e))
        
        # extract weights from each edge model
        model_weights = [t.result()["model_weights"] for t in tasks]
        
        if federated_mode == "average":
            average_weights = np.mean(model_weights, axis=0)

        elif federated_mode == "weighted_average":
            # get the weights
            sample_counts = np.array([t.result()["samples_count"] for t in tasks])
            edge_weights = get_edge_weights(sample_counts)
            
            # find weighted average
            average_weights = np.average(model_weights, weights=edge_weights, axis=0)

        else:
            raise Exception(f"Federated mode {federated_mode} is not recognized. \
                 Please select one of the available modes: ['average', 'weighted_average']")
            
        # assign the weights to the global_model
        global_model.set_weights(average_weights)

        print(f'Epoch {i}, Trained Federated Model')

        if x_test is not None and y_test is not None and evaluation_function and callable(evaluation_function):
            loss_eval, accuracy = evaluation_function(global_model, x_test, y_test)

        time.sleep(time_interval)

    return global_model


