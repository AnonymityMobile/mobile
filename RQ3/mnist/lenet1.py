import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
import pickle


def Lenet1():
    model = Sequential()
    # block1
    model.add(Conv2D(4, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), name='block1_pool1'))
    # block2
    model.add(Conv2D(12, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax', kernel_initializer='he_normal', name='dense_1'))
    sgd = SGD(lr=0.001, decay=0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


if __name__ == '__main__':

    batch_size = 128
    epochs = 6
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    y_test = to_categorical(y_test, 10)
    y_train = to_categorical(y_train, 10)

    print(len(x_train))
    print(len(x_test))

    model = Lenet1()
    model.fit(x_train, y_train, batch_size=batch_size, shuffle=True, epochs=epochs, validation_data=(x_test, y_test), verbose=1)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    model.save('lenet1.h5')


