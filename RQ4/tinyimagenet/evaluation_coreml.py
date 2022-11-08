import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def get_path_list(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if 'enhancement_prediction' in file_absolute_path and file_absolute_path.endswith('.pkl'):
                    path_list.append(file_absolute_path)
    return path_list


path_list = np.array(get_path_list('/Users/Documents/imagenet/mobile_model_data/tinyimagenet_prediction_resnet_coreml'))
y = pickle.load(open('/Users/Documents/imagenet/mobile_model_data/tinyimagenet_pkl/tiny_imagenet_label.pkl', 'rb'))
key_list = [i.split('/')[-1] for i in path_list]
print(len(key_list))



dic = {}
for i in range(len(path_list)):
    # prediction_np = pickle.load(open(path_list[i], 'rb'))['Identity']
    prediction = pickle.load(open(path_list[i], 'rb'))
    prediction_np = np.array([prediction[i]['Identity'][0] for i in range(len(prediction))])
    prediction_label_np = np.argmax(prediction_np, axis=1)

    prediction_label_np_train, prediction_label_np_test, y_train, y_test = train_test_split(prediction_label_np, y, test_size=0.2, random_state=5)

    # real_label_np = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_test, prediction_label_np_test)
    dic[key_list[i]] = acc

print(dic)