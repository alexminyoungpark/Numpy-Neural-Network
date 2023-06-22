import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    model = tf.keras.models.load_model("gen_num.h5")
    input_data = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
    prediction = model.predict(input_data)
    num = prediction.reshape((28, 28))
    plt.imsave("8.png", num, cmap="gray")
