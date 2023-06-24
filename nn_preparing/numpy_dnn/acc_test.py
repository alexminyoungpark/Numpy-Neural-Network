import numpy as np
import tensorflow as tf

def one_hot_encod(dataset):
    size = dataset.shape[-1]
    one_hot_shape = (size, 10)
    one_hot_y = np.zeros(one_hot_shape)
    one_hot_y[np.arange(size), dataset] = 1

    return one_hot_y.T


def flatten(dataset):
    size = dataset.shape[0]
    data_shape = (size, 784)

    return dataset.reshape(data_shape)


if __name__ == "__main__":
    ## mnist data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    ## process data
    x_train = flatten(x_train)
    x_test = flatten(x_test)
    y_test = one_hot_encod(y_test)
    y_train = one_hot_encod(y_train)

    model = tf.keras.models.load_model("./wbmodel.h5")
    weights = model.get_weights()
    w = weights[0]
    b = weights[1]
    b = np.reshape(b, (10,1))
    
    resA = np.dot(w.T, x_test.T) + b

    accuracy = 0
    for i, j in zip(resA.T, y_test.T):
        if np.argmax(i) == np.argmax(j):
            accuracy += 1

    print(f"accuracy: {accuracy/100}%")
