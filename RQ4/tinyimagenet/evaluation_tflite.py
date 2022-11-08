import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def get_path_list_densenet(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if 'enhancement_prediction_densenet' in file_absolute_path:
                    path_list.append(file_absolute_path)
    return path_list

def get_path_list_resnet(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if 'enhancement_prediction_resnet' in file_absolute_path:
                    path_list.append(file_absolute_path)
    return path_list


f = open('/mnt/irisgpfs/users/pycharm/data/tiny_imagenet_pkl/tiny_imagenet_label.pkl', 'rb')
y = pickle.load(f)

path_list = get_path_list_resnet('/mnt/irisgpfs/users/pycharm/data/tiny_imagenet_pkl')
print('len(path_list)', len(path_list))
key_list = [i.split('/')[-1] for i in path_list]

dic = {}
for i in range(len(path_list)):
    prediction_np = pickle.load(open(path_list[i], 'rb'))
    prediction_label_np = np.argmax(prediction_np, axis=1)
    prediction_label_np_train, prediction_label_np_test, y_train, y_test = train_test_split(prediction_label_np, y, test_size=0.2, random_state=5)
    acc = accuracy_score(y_test, prediction_label_np_test)
    dic[key_list[i]] = acc

print(dic)
