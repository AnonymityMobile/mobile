import tensorflow as tf
import numpy as np
import pickle


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


x = pickle.load(open('anything_image_data.pkl', 'rb'))
x = x.astype('float32')
if np.max(x) > 5:
    x = x / 255
x = x.reshape((len(x), len(x[0].flatten())))

model = get_auto_encoder(len(x[0].flatten()))
model.fit(x, x, epochs=50, batch_size=128, shuffle=True, validation_data=(x, x))
model.save('auto_encoder_anything.h5')



