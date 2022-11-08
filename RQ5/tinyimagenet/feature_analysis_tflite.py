from __future__ import print_function
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pickle
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from config_path import *
import argparse



def check(X):
    if np.max(X) > 5:
        X = X / 255.0
    return X


def get_statistic_feature(x):
    res = []
    for i in x:
        value_max_sub = i[np.argsort(i)[-1]] - i[np.argsort(i)[-2]]
        value_var = i.var()
        value_std = i.std()
        value_median = np.median(i)
        res.append([value_max_sub, value_var, value_std, value_median])
    res = np.array(res)
    return res


def get_graph_feature(x_prediction, path_tree_model, path_graph_model):
    tree = joblib.load(path_tree_model)
    tree_vec = pickle.load(open(path_graph_model, 'rb'))
    leaf_index_train = tree.apply(x_prediction)
    x_graph_vec = np.array([tree_vec[str(i)] for i in leaf_index_train])
    return x_graph_vec


def get_X_ypre(config_path_list, path_y):

    y = pickle.load(open(path_y, 'rb'))
    y = tf.keras.utils.to_categorical(y, num_classes)

    X_list_train = []
    ypre_list_train = []
    y_list_train = []

    X_list_test = []
    ypre_list_test = []
    y_list_test = []

    for i in config_path_list:
        print(i)

        tmp_X = check(pickle.load(open(i[0], 'rb')))
        tmp_ypre = np.array(pickle.load(open(i[1], 'rb')))

        tmp_x_train, tmp_x_test, y_train, y_test = train_test_split(tmp_X, y, test_size=0.2, random_state=5)
        tmp_ypre_train, tmp_y_pre_test, y_train, y_test = train_test_split(tmp_ypre, y, test_size=0.2, random_state=5)

        X_list_train.append(tmp_x_train)
        ypre_list_train.append(tmp_ypre_train)
        y_list_train.append(y_train)

        X_list_test.append(tmp_x_test)
        ypre_list_test.append(tmp_y_pre_test)
        y_list_test.append(y_test)


    x_train = np.concatenate(X_list_train, axis=0)
    x_train_prediction = np.concatenate(ypre_list_train, axis=0)
    y_train = np.concatenate(y_list_train, axis=0)

    x_test = np.concatenate(X_list_test, axis=0)
    x_test_prediction = np.concatenate(ypre_list_test, axis=0)
    y_test = np.concatenate(y_list_test, axis=0)

    return x_train, x_train_prediction, y_train, x_test, x_test_prediction, y_test

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--path_tree_model", type=str, default='')
ap.add_argument("-g", "--path_graph_model", type=str, default='')
ap.add_argument("-m", "--path_model", type=str, default='')
ap.add_argument("-f", "--path_feature_importance", type=str, default='')
args = vars(ap.parse_args())

path_tree_model = args['path_tree_model']
path_graph_model = args['path_graph_model']
model_path = args['path_model']
path_feature_importance = args['path_feature_importance']

x_train, x_train_prediction, y_train, x_test, x_test_prediction, y_test = get_X_ypre(path_pre_data_resnet_tflite_data, tiny_imagenet_label)

ai_model_output_dimension = 200
statistic_output_dimension = 4
graph_shape_output_dimension = 128

x_train_statistic_feature = get_statistic_feature(x_train_prediction)
x_train_graph_vec = get_graph_feature(x_train_prediction, path_tree_model, path_graph_model)

x_text_statistic_feature = get_statistic_feature(x_test_prediction)
x_test_graph_vec = get_graph_feature(x_test_prediction, path_tree_model, path_graph_model)


model = load_model(model_path)
print(model.summary())


middle_layer_model = Model(inputs=model.input, outputs=model.get_layer('concatenate').output)
middle_output = middle_layer_model.predict([x_train_prediction, x_train_statistic_feature, x_train_graph_vec, x_train])
print(middle_output.shape)

clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(middle_output, y_train)
feature_importance = clf.feature_importances_
print(feature_importance.shape)
pickle.dump(feature_importance, open(path_feature_importance, 'wb'), protocol=4)
print(feature_importance)

