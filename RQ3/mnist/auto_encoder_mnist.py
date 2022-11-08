import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist


def get_auto_encoder(n_neurons):
    input_img = tf.keras.Input(shape=(n_neurons,))
    encoded = tf.keras.layers.Dense(512, activation='relu')(input_img)
    encoded = tf.keras.layers.Dense(256, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(128, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(64, activation='relu', name='feature_vec')(encoded)
    decoded = tf.keras.layers.Dense(128, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(256, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(512, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(n_neurons, activation='sigmoid')(decoded)

    autoencoder = tf.keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.summary()
    return autoencoder


num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = x_train.reshape((len(x_train), len(x_train[0].flatten())))
x_test = x_test.reshape((len(x_test), len(x_test[0].flatten())))


model = get_auto_encoder(len(x_train[0].flatten()))
model.fit(x_train, x_train, epochs=100, batch_size=128, shuffle=True, validation_data=(x_test, x_test))
model.save('auto_encoder_mnist.h5')



