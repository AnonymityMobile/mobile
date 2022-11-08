from __future__ import print_function
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pickle
import numpy as np
import tensorflow as tf
import joblib
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

    X_list = []
    ypre_list = []
    y_list = []
    for i in config_path_list:
        print(i)
        tmp_X = check(pickle.load(open(i[0], 'rb')))
        tmp_ypre = np.array(pickle.load(open(i[1], 'rb')))
        X_list.append(tmp_X)
        ypre_list.append(tmp_ypre)
        y_list.append(y)
    X_np = np.concatenate(X_list, axis=0)
    ypre_np = np.concatenate(ypre_list, axis=0)
    y_np = np.concatenate(y_list, axis=0)

    return X_np, ypre_np, y_np

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

x_train, x_train_prediction, y_train = get_X_ypre(path_pre_data_vgg_tflite_train, 'y_train.pkl')
x_test, x_test_prediction, y_test = get_X_ypre(path_pre_data_vgg_tflite_test, 'y_test.pkl')

ai_model_output_dimension = 10
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

clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(middle_output, y_train)
feature_importance = clf.feature_importances_
print(feature_importance.shape)
pickle.dump(feature_importance, open(path_feature_importance, 'wb'), protocol=4)
print(feature_importance)
