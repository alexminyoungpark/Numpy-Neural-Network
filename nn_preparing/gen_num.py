import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    model = tf.keras.models.load_model("cce_gen_num.h5")

    label_0 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    label_1 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    label_2 = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
    label_3 = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
    label_4 = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    label_5 = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    label_6 = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
    label_7 = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
    label_8 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    label_9 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    img_0= model.predict(label_0).reshape((28, 28))
    img_1= model.predict(label_1).reshape((28, 28))
    img_2= model.predict(label_2).reshape((28, 28))
    img_3= model.predict(label_3).reshape((28, 28))
    img_4= model.predict(label_4).reshape((28, 28))
    img_5= model.predict(label_5).reshape((28, 28))
    img_6= model.predict(label_6).reshape((28, 28))
    img_7= model.predict(label_7).reshape((28, 28))
    img_8= model.predict(label_8).reshape((28, 28))
    img_9= model.predict(label_9).reshape((28, 28))

    plt.imsave("cce_0.png", img_0, cmap="gray")
    plt.imsave("cce_1.png", img_1, cmap="gray")
    plt.imsave("cce_2.png", img_2, cmap="gray")
    plt.imsave("cce_3.png", img_3, cmap="gray")
    plt.imsave("cce_4.png", img_4, cmap="gray")
    plt.imsave("cce_5.png", img_5, cmap="gray")
    plt.imsave("cce_6.png", img_6, cmap="gray")
    plt.imsave("cce_7.png", img_7, cmap="gray")
    plt.imsave("cce_8.png", img_8, cmap="gray")
    plt.imsave("cce_9.png", img_9, cmap="gray")
