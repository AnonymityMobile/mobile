import tensorflow as tf
import pickle
import coremltools
import numpy as np
from tensorflow.keras.datasets import cifar10


def get_shape_1_32_32_3(x):
    x = x.reshape(1, 32, 32, 3)
    x = x.astype(np.float32)
    return x

num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


def model_prediction(x_test, model, save_path):
    predict_list = []
    for x in x_test:
        x = get_shape_1_32_32_3(x)
        predict = list(model.predict({'conv2d_input': get_shape_1_32_32_3(x)})['Identity'][0])
        predict_list.append(predict)
    predict_np = np.array(predict_list)
    save_data = open(save_path, 'wb')
    pickle.dump(predict_np, save_data)


model = coremltools.models.MLModel('vgg.mlmodel')
model_prediction(x_train, model, 'cifar10_vgg_prediction_train_coreml.pkl')

