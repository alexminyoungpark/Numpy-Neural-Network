import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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


def main(
    batch_size: int = 1,
    epoch_size: int = 1,
) -> None:

    model = tf.keras.models.load_model(f"batch_experiment/model_60000_1000/model_60000_1000.h5")

#    label_0 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#    label_1 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
#    label_2 = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
#    label_3 = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
#    label_4 = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
#    label_5 = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
#    label_6 = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
#    label_7 = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
#    label_8 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
#    label_9 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    label_75 = np.array([[0, 0, 0, 0, 0, 1, 0, 1, 0, 0]])

#    img_0= model.predict(label_0).reshape((28, 28))
#    img_1= model.predict(label_1).reshape((28, 28))
#    img_2= model.predict(label_2).reshape((28, 28))
#    img_3= model.predict(label_3).reshape((28, 28))
#    img_4= model.predict(label_4).reshape((28, 28))
#    img_5= model.predict(label_5).reshape((28, 28))
#    img_6= model.predict(label_6).reshape((28, 28))
#    img_7= model.predict(label_7).reshape((28, 28))
#    img_8= model.predict(label_8).reshape((28, 28))
#    img_9= model.predict(label_9).reshape((28, 28))
    img_75= model.predict(label_75).reshape((28, 28))

#    plt.imsave(f"model_{batch_size}_{epoch_size}/{batch_size}_{epoch_size}_0.png", img_0, cmap="gray")
#    plt.imsave(f"model_{batch_size}_{epoch_size}/{batch_size}_{epoch_size}_1.png", img_1, cmap="gray")
#    plt.imsave(f"model_{batch_size}_{epoch_size}/{batch_size}_{epoch_size}_2.png", img_2, cmap="gray")
#    plt.imsave(f"model_{batch_size}_{epoch_size}/{batch_size}_{epoch_size}_3.png", img_3, cmap="gray")
#    plt.imsave(f"model_{batch_size}_{epoch_size}/{batch_size}_{epoch_size}_4.png", img_4, cmap="gray")
#    plt.imsave(f"model_{batch_size}_{epoch_size}/{batch_size}_{epoch_size}_5.png", img_5, cmap="gray")
#    plt.imsave(f"model_{batch_size}_{epoch_size}/{batch_size}_{epoch_size}_6.png", img_6, cmap="gray")
#    plt.imsave(f"model_{batch_size}_{epoch_size}/{batch_size}_{epoch_size}_7.png", img_7, cmap="gray")
#    plt.imsave(f"model_{batch_size}_{epoch_size}/{batch_size}_{epoch_size}_8.png", img_8, cmap="gray")
#    plt.imsave(f"model_{batch_size}_{epoch_size}/{batch_size}_{epoch_size}_9.png", img_9, cmap="gray")
    plt.imsave(f"75.png", img_75, cmap="gray")


if __name__ == "__main__":
#    kwargs = vars(parser.parse_args())
#    main(**kwargs)
    main(60000, 1000)
