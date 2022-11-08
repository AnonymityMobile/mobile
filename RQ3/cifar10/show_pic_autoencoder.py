import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10


def plot_some(im_list):
    plt.figure(figsize=(15, 4))
    for i, array in enumerate(im_list):
        plt.subplot(1, len(im_list), i+1)
        array = array.reshape(32, 32, 3)
        plt.imshow(array)
        plt.axis('off')
    plt.show()


num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = x_train.reshape((len(x_train), len(x_train[0].flatten())))
x_test = x_test.reshape((len(x_test), len(x_test[0].flatten())))


model = load_model('auto_encoder_cifar10.h5')

img_decoded = model.predict(x_train[10:20])

print('Before autoencoding:')
plot_some(x_train[10:20])
print('After decoding:')
plot_some(img_decoded)

