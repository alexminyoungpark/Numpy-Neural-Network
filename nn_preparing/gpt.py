import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.optimizers import Adam

# Load MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()

# Preprocess dataset
x_train = x_train / 255.0  # Normalize pixel values to [0, 1]
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)  # Convert labels to one-hot encoded vectors

# Model architecture
model = Sequential()
model.add(Dense(256, input_shape=(10,), activation='relu'))  # Input layer for labels
model.add(Dense(784, activation='sigmoid'))  # Output layer for number images
model.add(Reshape((28, 28)))  # Reshape output to image dimensions

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])

# Training
model.fit(y_train, x_train, batch_size=64, epochs=10)

## Inference
label0 = tf.keras.utils.to_categorical([0], num_classes=10)  # Example label (5)
label1 = tf.keras.utils.to_categorical([1], num_classes=10)  # Example label (5)
label2 = tf.keras.utils.to_categorical([2], num_classes=10)  # Example label (5)
generated_image0 = model.predict(label0)
generated_image1 = model.predict(label1)
generated_image2 = model.predict(label2)

# Visualize the generated image
import matplotlib.pyplot as plt
plt.imsave("gpt_0.png", generated_image0[0], cmap="gray")
plt.imsave("gpt_1.png", generated_image1[0], cmap="gray")
plt.imsave("gpt_2.png", generated_image2[0], cmap="gray")

