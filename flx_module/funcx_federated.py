# conda activate py37
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from funcx.sdk.client import FuncXClient
from funcx.sdk.executor import FuncXExecutor

def hello_world():
    print("Hello world!")

def get_data():
    from tensorflow import keras
    import numpy as np

    num_samples = 10

    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    
    # take a random set of images
    idx = np.random.choice(np.arange(len(x_train)), num_samples, replace=True)
    x_train = x_train[idx]
    y_train = y_train[idx]

    return (x_train, y_train)

def process_data(x_train, y_train):
    from tensorflow import keras
    import numpy as np

    num_classes = 10

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)

    return (x_train, y_train)

def train_model(json_model_config, 
                global_model_weights,
                x_train,
                y_train,
                batch_size=128,
                epochs=10,
                loss="categorical_crossentropy",
                optimizer="adam", 
                metrics=["accuracy"]):

    # import dependencies
    from tensorflow import keras

    # create the model
    model = keras.models.model_from_json(json_model_config)

    # compile the model and set weights to the global model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.set_weights(global_model_weights)

    # train the model on the local data and extract the weights
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model_weights = model.get_weights()

    return model_weights


def create_training_function(get_data=get_data, process_data=process_data, train_model=train_model):
    
    def training_function(json_model_config, 
                          global_model_weights):

        # import all the dependencies Irequired for funcX functions)
        import numpy as np

        # get data
        (x_train, y_train) = get_data()

        # process data
        (x_train, y_train) = process_data(x_train, y_train)

        model_weights = train_model(json_model_config, global_model_weights, x_train, y_train)
        np_model_weights = np.asarray(model_weights, dtype=object)

        return {"model_weights":np_model_weights, "samples_count": x_train.shape[0]}
    
    return training_function

def get_edge_weights(sample_counts):
    '''
    Returns weights for each model to find the weighted average 
    '''
    total = sum(sample_counts)
    fractions = sample_counts/total
    return fractions

def federated_average(global_model, endpoint_ids, get_data = get_data, process_data=process_data, train_model=train_model, weighted=False):
    fx = FuncXExecutor(FuncXClient())

    json_config = global_model.to_json()
    gm_weights = global_model.get_weights()
    gm_weights_np = np.asarray(gm_weights, dtype=object)

    # compile the training function
    training_function = create_training_function(get_data, process_data, train_model)
    
    # train the MNIST model on each of the endpoints and return the result, sending the global weights to each edge
    tasks = []
    for e in endpoint_ids:
        tasks.append(fx.submit(training_function, 
                                json_model_config=json_config, 
                                global_model_weights=gm_weights_np, 
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