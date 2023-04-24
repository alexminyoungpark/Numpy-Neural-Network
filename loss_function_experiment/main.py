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


def relu(Z):
    return np.maximum(0,Z)


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis = 0)


def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation == "relu":
        activation_func = relu
    elif activation == "softmax":
        activation_func = softmax
    else:
        raise Exception("Non-supported activation function")

    return activation_func(Z_curr), Z_curr


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

    ## set epsilon and make A1
    epsilon = 0.001
    A1 = np.copy(y_train)
    A1[A1 == 1] = 0.991
    A1[A1 == 0] = 0.001

    ## set Z1
    Z1 = np.log(A1)

    ## make W1. computing pseudo-inverse matrix
    W1 = np.dot(
        Z1,
        np.linalg.pinv(x_train.T)
    )
    b1 = np.zeros((10, 1))

    ## get accuracy
    resA, resB = single_layer_forward_propagation(x_test.T, W1, b1, activation="softmax")
    print(resA.shape)
    accuracy = 0
    for i, j in zip(resA.T, y_test.T):
        if np.argmax(i) == np.argmax(j):
            accuracy += 1

    print(f"accuracy: {accuracy/100}%")
