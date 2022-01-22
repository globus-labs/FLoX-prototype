from funcx_federated import create_inference_function, federated_decorator, create_training_function

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def get_test_data(num_samples=1000):
    num_classes = 10
    _, (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    idx = np.random.choice(np.arange(len(x_test)), num_samples, replace=True)
    x_test = x_test[idx]
    y_test = y_test[idx]
    
    x_test = x_test.astype("float32") / 255
    x_test = np.expand_dims(x_test, -1)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_test, y_test)

def get_train_data(num_samples=1000):
    num_classes = 10
    (x_train, y_train), _= keras.datasets.fashion_mnist.load_data()
    
    idx = np.random.choice(np.arange(len(x_train)), num_samples, replace=True)
    x_train = x_train[idx]
    y_train = y_train[idx]
    
    x_train = x_train.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    return (x_train, y_train)

def eval_model(m, x, y):
    ''' evaluate model on dataset x,y'''
    score = m.evaluate(x, y, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

def preprocess_data(x_train, y_train, num_samples=100):
    from tensorflow import keras
    import numpy as np

    num_classes = 10

    # take a random set of images
    idx = np.random.choice(np.arange(len(x_train)), num_samples, replace=True)
    x_train = x_train[idx]
    y_train = y_train[idx]

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)

    return (x_train, y_train)

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