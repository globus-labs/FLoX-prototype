import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time
from timeit import default_timer as timer
import csv

from funcx.sdk.client import FuncXClient
from funcx.sdk.executor import FuncXExecutor

def federated_decorator(func):
    """
    Returns a wrapped function that can be deployed to funcx endpoints when given endpoint_ids

    Parameters
    ----------
    func: function
        the function to be deployed to funcx endpoints

    Returns
    -------
    wrapper: function
        the wrapped function func
    
    """
    def wrapper(*args, **kwargs):
        """
        Returns a wrapped function that is deployed to specified funcx endpoints.

        Parameters
        ----------
        *args: positional arguments

        **kwargs: keyword arguments
            when using a wrapped function, you need to pass in parameters as 
            keywords

        Returns
        -------
        tasks: list
            contains funcx task objects from which the original result can be retrieved with
            task[i].result()

        """
        fx = FuncXExecutor(FuncXClient())
        tasks = []

        # for each endpoint, submit the function with **kwargs to it
        for e in kwargs["endpoint_ids"]:
            tasks.append(fx.submit(func, 
                                   *args,
                                   **kwargs,
                                    endpoint_id=e))
        return tasks
    
    return wrapper

# path_dir='/home/pi/datasets', x_train_path="mnist_x_train.npy", y_train_path="mnist_y_train.npy"
def get_local_data(x_train_name, y_train_name, path_dir=".", preprocess=False, preprocessing_function=None):
    """
    Retrieves a local dataset given the path and filenames. Optionally, also processes the data
    with the given processing_function.

    Parameters
    ----------
    x_train_name: str
        filename specifying x_train

    y_train_name: str
        filename specifying y_train

    path_dir: str
        path to the folder where x_train_name and y_train_name reside

    preprocess: boolean
        if True, calls preprocessing_function on (x_train, y_train)

    preprocessing_function: function
        a custom user-provided function that processes (x_train, y_train) and outputs 
        a tuple (x_train, y_train)

    Returns
    -------
    (x_train, y_train): tuple
        returns the loaded (and optionally preprocessed) (x_train, y_train)
    
    
    """
    import numpy as np
    import os
    
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
        if not preprocessing_function or not callable(preprocessing_function):
            raise TypeError('preprocessing_function is not a function. \
                            Please provide a valid function in your call')
        
        (x_train, y_train) = preprocessing_function(x_train, y_train)

    return (x_train, y_train)


def get_keras_data(keras_dataset='mnist', num_samples=None, preprocess=True, preprocessing_function=None, **kwargs):
    """
    Returns (x_train, y_train) of a chosen built-in Keras dataset. 
    Also preprocesses the image datasets (mnist, fashion_mnist, cifar10, cifar100) by default.

    Parameters
    ----------
    keras_dataset: str
        one of the available Keras datasets: 
        ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'reuters', 'boston_housing']

    num_samples: int 
        randomly samples n data points from (x_train, y_train). Set to None by default.

    preprocess: boolean
        if True, preprocesses (x_train, y_train) 

    preprocessing_function: function
        a custom user-provided function that processes (x_train, y_train) and outputs 
        a tuple (x_train, y_train)

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
        (x_train, y_train), _ = dataset_mapping[keras_dataset].load_data()

        # take a random set of images
        if num_samples:
            idx = np.random.choice(np.arange(len(x_train)), num_samples, replace=True)
            x_train = x_train[idx]
            y_train = y_train[idx]

        if preprocess:
            if preprocessing_function and callable(preprocessing_function):
                (x_train, y_train) = preprocessing_function(x_train, y_train)

            else:
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

        return (x_train, y_train)

def train_default_model(json_model_config, 
                global_model_weights,
                x_train,
                y_train,
                epochs=10,
                input_shape=(32, 28, 28, 1),
                loss="categorical_crossentropy",
                optimizer="adam", 
                metrics=["accuracy"]):
    """
    Conctructs, compiles, and trains a Tensorflow model with provided training data

    Parameters
    ----------
    json_model_config: str
        configuration of the TF model retrieved using model.to_json()

    global_model_weights: numpy array
        a numpy array with weights of the TF model

    (x_train, y_train): dataset
        a suitable format for TF's model.fit() function

    epochs: int
        the number of epochs to train the model for

    (loss, optimizer, metrics): 
        values for TF's model.compile() function

    Returns
    -------
    np_model_weights: numpy array
        numpy array with updated weights of the TF model

    Notes
    -----
    This function does not support more arguments for .fit() or .compile().
    The fix is as easy as just exposing these parameters within the function.
    However, there are way too many methods for training models so instead we 
    invite users to create custom training functions for more complex examples
    
    """

    # import dependencies
    from tensorflow import keras
    import numpy as np

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
        # change the INPUT SHAPE! make it dynamic
        model.build(input_shape=input_shape)
        model.set_weights(global_model_weights)

    # train the model on the local data and extract the weights
    model.fit(x_train, y_train, epochs=epochs)
    model_weights = model.get_weights()

    # transform to a numpy array
    np_model_weights = np.asarray(model_weights, dtype=object)

    return np_model_weights


def create_training_function(train_model=train_default_model, 
                            data_source: str = "keras",
                            path_dir='/home/pi/datasets', 
                            x_train_name="mnist_x_train.npy", 
                            y_train_name="mnist_y_train.npy", 
                            preprocess=False, 
                            preprocessing_function=None,
                            keras_dataset = "mnist", 
                            input_shape=(32, 28, 28, 1),
                            loss="categorical_crossentropy",
                            optimizer="adam", 
                            metrics=["accuracy"],
                            get_keras_data=get_keras_data,
                            get_local_data=get_local_data,
                            get_custom_data=None,
                            **kwargs
):
    """
    Creates a function for loading data and training a model

    Parameters
    ----------
    data_source: str
        the function supports three data sources: "local", "keras", "custom"
        for "local" and "keras", see get_local_data and get_keras_data functions
        "custom" is for a user-provided data-retrieving function

    path_dir: str
        needed when data_sourse="local"; path to x_train and y_train filenames

    x_train_name: str
        needed when data_sourse="local"; filename for x_train

    y_train_name: str
        needed when data_sourse="local"; file name for y_train

    preprocess: boolean
        if True, will attempt to preprocess your data in "local" or "keras" data sources
        see get_local_data and get_keras_data functions

    preprocessing_function: function
        user-provided function for processing data in "local" or "keras" data sources

    keras_dataset: str
        specifies one of the default keras datasets to use if using "keras" data source
        ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'reuters', 'boston_housing']

    loss: str
        loss for TF's model.fit() function

    optimizer: str
        optimizer for for TF's model.fit() function

    metrics: str/list 
        metrix for TF's model.fit() function. E.g, metrics=["accuracy"],

    get_keras_data: function
        default function for get_keras_data

    get_local_data: function
        default function for get_keras_data

    get_custom_data: function
        user-provided function for retrieving data

    Returns
    -------
    Function for retrieving, processing, and training a Tensorflow model

    Notes
    -----
    This function is aimed at simple use cases. 
    You can construct & easily use any custom training_function with funcX 
    
    """    
    def training_function(json_model_config, 
                          global_model_weights, 
                          num_samples=None,
                          epochs=10,
                          **kwargs
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
        from timeit import default_timer as timer
        task_start = timer()
        # import all the dependencies required for funcX functions)
        import numpy as np

        # retrieve (and optionally process) the data
        if data_source == 'local':
            (x_train, y_train) = get_local_data(path_dir=path_dir, 
                          x_train_name=x_train_name, 
                          y_train_name=y_train_name, 
                          preprocess=preprocess, 
                          preprocessing_function=preprocessing_function)

        elif data_source == 'keras':
            (x_train, y_train) = get_keras_data(keras_dataset=keras_dataset, 
                                                preprocess=preprocess, 
                                                num_samples=num_samples,
                                                preprocessing_function=preprocessing_function)

        elif data_source == 'custom':
            if callable(get_custom_data):
                (x_train, y_train) = get_custom_data()
            else:
                raise TypeError('preprocessing_function is not a function. \
                                 Please provide a valid function in your call')

        else:
            raise Exception("Please choose one of data sources: ['local', 'keras', 'custom']")

        # train the model
        training_start = timer()
        model_weights = train_model(json_model_config=json_model_config, 
                                            global_model_weights=global_model_weights, 
                                            x_train=x_train, 
                                            y_train=y_train,
                                            epochs=epochs,
                                            input_shape=input_shape,
                                            loss=loss,
                                            optimizer=optimizer, 
                                            metrics=metrics)

        training_runtime = timer() - training_start
        task_runtime = timer() - task_start
        # return the updated weights and number of samples the model was trained on
        return {"model_weights":model_weights, "samples_count": x_train.shape[0], 'task_runtime':task_runtime, 'training_runtime': training_runtime}
    
    return training_function

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
    ''' evaluate model on dataset x,y'''
    score = m.evaluate(x, y, verbose=0)
    if not silent:
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    return score[0], score[1]

def find_federated_average(model_weights, weighted=False, calculation_weights=None):
    if weighted:
        # find weighted average
        average_weights = np.average(model_weights, weights=calculation_weights, axis=0)
        
    else:
        # simple average of the weights
        average_weights = np.mean(model_weights, axis=0)

    return average_weights

def federated_learning(global_model, 
                      endpoint_ids, 
                      num_samples=100,
                      epochs=10,
                      loops=1,
                      time_interval=0,
                      federated_mode="average",
                      train_model=train_default_model, 
                      data_source: str = "keras",
                      path_dir='/home/pi/datasets', 
                      x_train_name="mnist_x_train.npy", 
                      y_train_name="mnist_y_train.npy", 
                      preprocess=False, 
                      preprocessing_function=None,
                      keras_dataset = "mnist",
                      input_shape=(32, 28, 28, 1),  
                      loss="categorical_crossentropy",
                      optimizer="adam", 
                      metrics=["accuracy"],
                      get_custom_data=None,
                      evaluation_function=eval_model,
                      x_test=None,
                      y_test=None,
                      csv_path='/content/drive/MyDrive/flx/evaluation/experiments.csv',
                      experiment='default',
                      description='default',
                      dataset_name="mnist",
                      client_names="RPi4-8gb, RPi4-4gb",
                      **kwargs):
    """
    TODO: seems like this function is redundant and can be replaced with a sequence
    of a few lower-level functions.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    
    """
    fx = FuncXExecutor(FuncXClient())

    # compile the training function
    training_function = create_training_function(train_model=train_model, 
                                                data_source = data_source,
                                                path_dir=path_dir, 
                                                x_train_name=x_train_name, 
                                                y_train_name=y_train_name, 
                                                preprocess=preprocess, 
                                                preprocessing_function=preprocessing_function,
                                                keras_dataset = keras_dataset,
                                                input_shape=input_shape, 
                                                loss=loss,
                                                optimizer=optimizer, 
                                                metrics=metrics,
                                                get_custom_data=get_custom_data)
    
    federated_training = federated_decorator(training_function)
    
    for i in range(loops):
        round_start = timer()
        # get the model's architecture and weights
        json_config = global_model.to_json()
        gm_weights = global_model.get_weights()
        gm_weights_np = np.asarray(gm_weights, dtype=object)

        tasks_start = timer()
        # train the MNIST model on each of the endpoints and return the result, sending the global weights to each edge
        tasks = federated_training(json_model_config=json_config, 
                                    global_model_weights=gm_weights_np, 
                                    num_samples=num_samples,
                                    epochs=epochs,
                                    endpoint_ids=endpoint_ids)
        
        # extract weights from each edge model
        model_weights = [t.result()["model_weights"] for t in tasks]
        tasks_sending_runtime = timer() - tasks_start

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
                 Please select on of the available modes: ['average', 'weighted_average']")
            
        # assign the weights to the global_model
        global_model.set_weights(average_weights)

        print(f'Epoch {i}, Trained Federated Model')

        if x_test is not None and y_test is not None and evaluation_function and callable(evaluation_function):
            loss_eval, accuracy = evaluation_function(global_model, x_test, y_test)

        #time.sleep(time_interval)

        round_runtime = timer() - round_start


        endpoint_task_runtimes = [t.result()["task_runtime"] for t in tasks]
        average_task_runtime = np.mean(endpoint_task_runtimes, axis=0)
        endpoint_task_runtimes = [round(i, 3) for i in endpoint_task_runtimes]

        endpoint_training_runtimes = [t.result()["training_runtime"] for t in tasks]
        average_training_runtime = np.mean(endpoint_training_runtimes, axis=0)
        endpoint_training_runtimes = [round(i, 3) for i in endpoint_training_runtimes]
        
        communication_time = tasks_sending_runtime - max(endpoint_task_runtimes)

        model_size = sum(w.size for w in gm_weights_np) * gm_weights_np.itemsize

        endpoint_losses = []
        endpoint_accuracies = []

        for m_weight in model_weights:
            m = keras.models.model_from_json(json_config)
            m.compile(loss=loss, optimizer=optimizer, metrics=metrics)    
            try:
                m.set_weights(m_weight)
            except:
                m.build(input_shape=input_shape)
                m.set_weights(m_weight)

            e_loss, e_accuracy = evaluation_function(m, x_test, y_test, silent=True)
            endpoint_losses.append(round(e_loss, 3))
            endpoint_accuracies.append(round(e_accuracy, 3))

        if data_source == "keras":
            dataset_name = keras_dataset

        header = ['experiment', 'description', 'round', 'epochs', 'num_samples', 'dataset', 'n_clients',
         'accuracy', 'endpoint_accuracies', 'loss', 'endpoint_losses', 'round_runtime',
          'task_and_sending_runtime', 'average_task_runtime',  'endpoint_task_runtimes',
           'communication_time', 'average_training_runtime', 'endpoint_training_runtimes',
            'client_names', 'files_size']

        csv_entry = {'experiment':experiment,
                    'description':description,
                    'round':i, 
                    'epochs':epochs, 
                    'num_samples':num_samples, 
                    'dataset':dataset_name, 
                    'n_clients':len(endpoint_ids), 
                    'accuracy':accuracy,
                    'endpoint_accuracies': endpoint_accuracies, 
                    'loss':loss_eval,
                    'endpoint_losses': endpoint_losses, 
                    'round_runtime':round_runtime, 
                    'task_and_sending_runtime':tasks_sending_runtime,
                    'average_task_runtime': average_task_runtime,
                    'endpoint_task_runtimes': endpoint_task_runtimes,
                    'communication_time': communication_time,
                    'average_training_runtime': average_training_runtime,
                    'endpoint_training_runtimes': endpoint_training_runtimes,
                    'client_names':client_names,
                    'files_size':model_size}

        with open(csv_path, 'a', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, header)
            writer.writerow(csv_entry)

    return global_model

def store_inference_results(probs, path_dir='/home/pi/globus', filename='results.csv'):
    """
    Stores results of model.predict() in a csv file locally.

    Parameters
    ----------
    probs: numpy array
        output of model.predict()

    path_dir: str
        path to the directory of the CSV filename (where to store the file)

    filename: str
        name of the CSV file

    Notes
    -----
    The function stores this information about the given array:
    
    - timestamp: timestamp of when the information was added to the file. 
        this can be used to track when the inference was made

    - value_counts: number of appearances of each class
    
    """
    from datetime import datetime
    import csv
    import os
    import numpy as np

    # get the path
    results_file = os.sep.join([path_dir, filename])

    # extract the likeliest class based on the probability vector
    predictions = np.argmax(probs, axis=1)

    # count the occurances of classes
    unique, counts = np.unique(predictions, return_counts=True)
    unique = unique.tolist()
    counts = counts.tolist()

    pred_counts = dict(zip(unique, counts))

    # get the timestamp
    timestamp = datetime.now()
    date_time = timestamp.strftime("%m/%d/%Y, %H:%M:%S")

    # add the information to the csv file
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow([date_time, pred_counts])

def retrieve_inference_results(path_dir='/home/pi/globus', filename='results.csv', **kwargs):
    """
    Returns information stored in the csv file as a list of rows

    Parameters
    ----------
    path_dir: str
        path to the directory of the CSV filename (where to store the file)

    filename: str
        name of the CSV file

    Returns
    -------
    rows: list
        a list of rows from the CSV file

    """
    import os
    import csv
    
    # construct the path
    results_file = os.sep.join([path_dir, filename])
    
    rows = []
    with open(results_file, 'r') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            rows.append(row)
        
    return rows

def create_inference_function(data_source: str = "keras",
                            path_dir='/home/pi/datasets', 
                            x_train_name="mnist_x_train.npy", 
                            y_train_name="mnist_y_train.npy", 
                            preprocess=False, 
                            preprocessing_function=None,
                            keras_dataset = "mnist", 
                            loss="categorical_crossentropy",
                            optimizer="adam", 
                            metrics=["accuracy"],
                            get_keras_data=get_keras_data,
                            get_local_data=get_local_data,
                            get_custom_data=None,
                            store_results=store_inference_results,
                            store_results_path='/home/pi/globus', 
                            filename='results.csv',
                            **kwargs
):
    """
    Creates a function for loading data and doing inference on it with a Tensorflow model

    Parameters
    ----------
    data_source: str
        the function supports three data sources: "local", "keras", "custom"
        for "local" and "keras", see get_local_data and get_keras_data functions
        "custom" is for a user-provided data-retrieving function

    path_dir: str
        needed when data_sourse="local"; path to x_train and y_train filenames

    x_train_name: str
        needed when data_sourse="local"; filename for x_train

    y_train_name: str
        needed when data_sourse="local"; file name for y_train

    preprocess: boolean
        if True, will attempt to preprocess your data in "local" or "keras" data sources
        see get_local_data and get_keras_data functions

    preprocessing_function: function
        user-provided function for processing data in "local" or "keras" data sources

    keras_dataset: str
        specifies one of the default keras datasets to use if using "keras" data source
        ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'reuters', 'boston_housing']

    loss: str
        loss for TF's model.fit() function

    optimizer: str
        optimizer for for TF's model.fit() function

    metrics: str/list 
        metrix for TF's model.fit() function. E.g, metrics=["accuracy"],

    get_keras_data: function
        default function for get_keras_data

    get_local_data: function
        default function for get_keras_data

    get_custom_data: function
        user-provided function for retrieving data

    Returns
    -------
    Function for retrieving, processing, and training a Tensorflow model

    Notes
    -----
    This function is aimed at simple use cases. 
    You can construct & easily use any custom inference_function with funcX 
    
    """   
    def inference_function(json_model_config, 
                          global_model_weights, 
                          num_samples=None,
                          loops=2,
                          time_interval=0,
                          **kwargs
):
        """
        A function for doing inference at the edge. 
        Loads & preprocesses the data, constructs & compiles the model
        Does so in a for-loop with a chosen time_interval

        Parameters
        ----------
        json_model_config: str
        configuration of the TF model retrieved using model.to_json()

        global_model_weights: numpy array
            a numpy array with weights of the TF model

        num_samples: int
            if data_source="keras", randomly samples n data points from (x_train, y_train)

        loops: int
            the number of loops to run the workflow for

        time_interval: int 
            number of seconds to wait for before executing the next data-gathering and inference loop

        Returns
        -------
        TODO: return stats about the inference process

        """
        # import dependencies
        from tensorflow import keras
        import numpy as np
        import time

        # create the model
        model = keras.models.model_from_json(json_model_config)

        # compile the model and set weights to the global model
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        try:
            model.set_weights(global_model_weights)
        except:
            model.build(input_shape=(32, 28, 28, 1))
            model.set_weights(global_model_weights)

        #model.set_weights(global_model_weights)

        for i in range(loops):
        # train the model on the local data and extract the weights

            if data_source == 'local':
                (x_train, y_train) = get_local_data(path_dir=path_dir, 
                            x_train_name=x_train_name, 
                            y_train_name=y_train_name, 
                            preprocess=preprocess, 
                            preprocessing_function=preprocessing_function)

            elif data_source == 'keras':
                (x_train, y_train) = get_keras_data(keras_dataset=keras_dataset, 
                                                    preprocess=preprocess, 
                                                    num_samples=num_samples)

            elif data_source == 'custom':
                if callable(get_custom_data):
                    (x_train, y_train) = get_custom_data()
                else:
                    raise TypeError('preprocessing_function is not a function. Please provide a valid function in your call')

            else:
                raise Exception("Please choose one of data sources: ['local', 'keras', 'pass']")

            predictions = model.predict(x_train)

            store_results(probs=predictions, path_dir=store_results_path, filename=filename)

            # wait for time_interval seconds 
            time.sleep(time_interval)

            # save the stats in a file for future retrieval

        return "The inference process is over"
    
    return inference_function

def get_keras_data_timed(keras_dataset='mnist', num_samples=None, preprocess=True, preprocessing_function=None, **kwargs):
    """
    Returns (x_train, y_train) of a chosen built-in Keras dataset. 
    Also preprocesses the image datasets (mnist, fashion_mnist, cifar10, cifar100) by default.

    Parameters
    ----------
    keras_dataset: str
        one of the available Keras datasets: 
        ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'reuters', 'boston_housing']

    num_samples: int 
        randomly samples n data points from (x_train, y_train). Set to None by default.

    preprocess: boolean
        if True, preprocesses (x_train, y_train) 

    preprocessing_function: function
        a custom user-provided function that processes (x_train, y_train) and outputs 
        a tuple (x_train, y_train)

    Returns
    -------

    
    """
    from timeit import default_timer as timer
    task_start = timer()
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

        if preprocess:
            if preprocessing_function and callable(preprocessing_function):
                (x_train, y_train) = preprocessing_function(x_train, y_train)

            else:
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

        task_time = timer() - task_start

        return {'dataset':(x_train, y_train), 'runtime': task_time}
