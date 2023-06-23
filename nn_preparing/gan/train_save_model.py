import argparse
import numpy as np
import tensorflow as tf

#parser = argparse.ArgumentParser(
#    prog=__name__.rsplit(".", 1)[-1],
#    formatter_class=argparse.RawTextHelpFormatter,
#)
#parser.add_argument(
#    "--batch-size",
#    default=1,
#    type=int,
#)
#parser.add_argument(
#    "--epoch-size",
#    default=1,
#    type=int,
#)

def one_hot_encod(dataset):
    size = dataset.shape[-1]
    one_hot_shape = (size, 10)
    one_hot_y = np.zeros(one_hot_shape)
    one_hot_y[np.arange(size), dataset] = 1

    return one_hot_y

def flatten(dataset):
    size = dataset.shape[0]
    data_shape = (size, 784)

    return dataset.reshape(data_shape)

def main(
    batch_size: int = 1,
    epoch_size: int = 1,
) -> None:
    ## mnist data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    ## process data
    x_train = flatten(x_train)
    x_test = flatten(x_test)
    y_test = one_hot_encod(y_test)
    y_train = one_hot_encod(y_train)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(25, input_shape=(10, ), activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(784, activation='sigmoid'),
    ])

    loss_function = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    model.fit(y_train, x_train, epochs=1000, batch_size=60000)

    test_loss, test_acc = model.evaluate(y_test, x_test)
    print('Test accuracy: ', test_acc)
    model.save(f'./model_60000_100.h5')


if __name__ == "__main__":
#    kwargs = vars(parser.parse_args())
#    main(**kwargs)
    main()
