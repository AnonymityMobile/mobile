from __future__ import print_function

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model


def cnn(x_train, num_classes):

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, name='dense_1'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    batch_size = 64
    epochs = 80
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # pickle.dump(x_train, open('/Users/Documents/imagenet/cifar10/x_train.pkl', 'wb'))
    # pickle.dump(x_test, open('/Users/Documents/imagenet/cifar10/x_test.pkl', 'wb'))
    # pickle.dump(y_train, open('/Users/Documents/imagenet/cifar10/y_train.pkl', 'wb'))
    # pickle.dump(y_test, open('/Users/Documents/imagenet/cifar10/y_test.pkl', 'wb'))

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # model = cnn(x_train, num_classes=10)
    # model.summary()
    # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
    #
    # scores = model.evaluate(x_test, y_test, verbose=0)
    # print("Accuracy: %.2f%%" % (scores[1] * 100))
    # model.save('cnn.h5')
    # model.summary()

    model = load_model('vgg.h5')
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))   # Accuracy: 80.06 %


