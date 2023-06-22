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
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

# Training
model.fit(y_train, x_train, batch_size=64, epochs=10, validation_split=0.2)

# Inference
label = tf.keras.utils.to_categorical([5], num_classes=10)  # Example label (5)
generated_image = model.predict(label)

# Visualize the generated image
import matplotlib.pyplot as plt
plt.imsave("5.png", generated_image[0], cmap="gray")

