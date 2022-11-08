from __future__ import print_function
import tensorflow as tf
import pickle
import numpy as np
import joblib
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from sklearn import tree
from sklearn.metrics import accuracy_score

path_train = '/Users/Documents/imagenet/mobile_model_data/RQ1/prediction/cifar10_cnn_prediction_train_coreml.pkl'
path_test = '/Users/Documents/imagenet/mobile_model_data/RQ1/prediction/cifar10_cnn_prediction_test_coreml.pkl'
save_tree_model = 'cifar10_cnn_tree_coreml.model'
save_tree_path = 'cifar10_cnn_tree_coreml.txt'


def get_tree_structure(save_tree_path, x, model):
    path_tree = model.decision_path(x).toarray()
    data_path_res = []
    for path in path_tree:
        res = [i for i in range(len(path)) if path[i] == 1]
        data_path_res.append(res)

    tree_edge = []
    for path in data_path_res:
        for i in range(1, len(path)):
            edge = [path[i-1], path[i]]
            tree_edge.append(edge)

    tree_edge = [list(t) for t in set(tuple(_) for _ in tree_edge)]

    re = open(save_tree_path, 'a')
    for edge in tree_edge:
        re.write(str(edge[0]) + ' ' + str(edge[1]) + '\n')
    re.close()


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


x_train_prediction = np.array(pickle.load(open(path_train, 'rb')))
x_test_prediction = np.array(pickle.load(open(path_test, 'rb')))
model = tree.DecisionTreeClassifier(max_depth=15)
model.fit(x_train_prediction, y_train)
joblib.dump(model, save_tree_model)

y_test_pre = model.predict(x_test_prediction)
y_train_pre = model.predict(x_train_prediction)
print(accuracy_score(y_test, y_test_pre))
print(accuracy_score(y_train, y_train_pre))
all_x = np.concatenate((x_train_prediction, x_test_prediction), axis=0)
get_tree_structure(save_tree_path, all_x, model)






