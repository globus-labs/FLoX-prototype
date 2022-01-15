

def process_data(x_train, y_train, num_samples=100):
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