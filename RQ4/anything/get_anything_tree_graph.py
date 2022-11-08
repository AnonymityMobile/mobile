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
from sklearn.model_selection import train_test_split

save_tree_model = 'anything_tree.model'
save_tree_path = 'anything_tree.txt'


dic_mapping = {'n01872401': 103, 'n01806143': 85, 'n04254680': 806, 'n04254777': 807, 'n04355933': 837,
               'n04371774': 844, 'n04483307': 872, 'n04507155': 880, 'n04532670': 889, 'n04552348': 896}
n_vec = 1001


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


def get_y_vec(dic_mapping, n_vec, y):
    y_int = [dic_mapping[i] for i in y]
    y_vec = []
    for i in y_int:
        tmp_vec = [0]*n_vec
        tmp_vec[i] = 1
        y_vec.append(tmp_vec)
    y_vec = np.array(y_vec)
    return y_vec


y = pickle.load(open('anything_image_label.pkl', 'rb'))
y_label_vec = get_y_vec(dic_mapping, n_vec, y)

prediction_np = np.array(pickle.load(open('anything_prediction_test.pkl', 'rb')))

x_train_prediction, x_test_prediction, y_train, y_test = train_test_split(prediction_np, y, test_size=0.2, random_state=5)

model = tree.DecisionTreeClassifier(max_depth=15)
model.fit(x_train_prediction, y_train)
joblib.dump(model, save_tree_model)

y_test_pre = model.predict(x_test_prediction)
y_train_pre = model.predict(x_train_prediction)
print(accuracy_score(y_test, y_test_pre))
print(accuracy_score(y_train, y_train_pre))
all_x = np.concatenate((x_train_prediction, x_test_prediction), axis=0)
get_tree_structure(save_tree_path, all_x, model)








