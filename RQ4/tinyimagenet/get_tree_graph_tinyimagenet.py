from __future__ import print_function
import tensorflow as tf
import pickle
import numpy as np
import joblib
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from sklearn import tree
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


save_tree_model = 'tinyimagenet_resnet_coreml.model'
save_tree_path = 'tinyimagenet_resnet_coreml.txt'


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


X = pickle.load(open('tiny_imagenet_data.pkl', 'rb'))
y = pickle.load(open('tiny_imagenet_label.pkl', 'rb'))

num_classes = 200
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

x_train_prediction = pickle.load(open('tinyimagenet_resnet_prediction_train_coreml.pkl', 'rb'))
x_test_prediction = pickle.load(open('tinyimagenet_resnet_prediction_test_coreml.pkl', 'rb'))

model = tree.DecisionTreeClassifier(max_depth=15)
model.fit(x_train_prediction, y_train)
joblib.dump(model, save_tree_model)

y_test_pre = model.predict(x_test_prediction)
y_train_pre = model.predict(x_train_prediction)
print(accuracy_score(y_test, y_test_pre))
print(accuracy_score(y_train, y_train_pre))
all_x = np.concatenate((x_train_prediction, x_test_prediction), axis=0)
get_tree_structure(save_tree_path, all_x, model)


