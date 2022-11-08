import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import accuracy_score


def get_path_list_cnn(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if 'enhancement_prediction_cnn' in file_absolute_path:
                    path_list.append(file_absolute_path)
    return path_list

def get_path_list_vgg(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if 'enhancement_prediction_vgg' in file_absolute_path:
                    path_list.append(file_absolute_path)
    return path_list

num_classes=10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = x_train.reshape((len(x_train), len(x_train[0].flatten())))
x_test = x_test.reshape((len(x_test), len(x_test[0].flatten())))


path_list = get_path_list_vgg('/mnt/irisgpfs/users/pycharm/data/cifar10_pkl')
key_list = [i.split('/')[-1] for i in path_list]

dic = {}
for i in range(len(path_list)):
    prediction_np = pickle.load(open(path_list[i], 'rb'))
    prediction_label_np = np.argmax(prediction_np, axis=1)
    real_label_np = np.argmax(y_test, axis=1)
    acc = accuracy_score(real_label_np, prediction_label_np)
    dic[key_list[i]] = acc

print(dic)
