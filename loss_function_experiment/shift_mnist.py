import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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


def shift_and_pad_image(image, num):
    shifted_image = np.zeros_like(image)
    shifted_image[num:, num:] = image[:-num, :-num]
    return shifted_image


if __name__ == "__main__":
    ## mnist data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    plt.imsave("original.png", x_train[0], cmap="gray")

    for idx, img in enumerate(x_train):
        x_train[idx] = shift_and_pad_image(img, 2)

    plt.imsave("shift.png", x_train[0], cmap="gray")

    ## process data
#    x_train = flatten(x_train)
#    x_test = flatten(x_test)
#    y_test = one_hot_encod(y_test)
#    y_train = one_hot_encod(y_train)

    A1 = np.copy(y_train)

#    w = np.load("./pinv.npy")
#    shift_w = np.dot(
#        A1,
#        np.linalg.pinv(x_train.T)
#    )

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.0)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=10, batch_size=600)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test accuracy", test_acc)

#    weights = model.get_weights()
#    w = weights[0]
#    b = weights[1]
#    b = np.reshape(b, (10, 1))
#
#    resA = np.dot(w.T, x_test.T) + b
#
#    accuracy = 0
#    for i, j in zip(resA.T, y_test.T):
#        if np.argmax(i) == np.argmax(j):
#            accuracy += 1
#
#    print(f"accuracy: {accuracy/100}%")
