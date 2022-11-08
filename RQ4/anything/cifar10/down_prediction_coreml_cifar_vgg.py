import tensorflow as tf
import numpy as np
import argparse
import pickle
import coremltools
from tensorflow.keras.datasets import cifar10
import joblib


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

def check_X(X):
    if np.max(X) > 10:
        X = X / 255.0
    else:
        X = X
    return X


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pkl_path", type=str, default='')
ap.add_argument("-s", "--save_path", type=str, default='')
ap.add_argument("-d", "--prediction_path", type=str, default='')
args = vars(ap.parse_args())
pkl_path = args['pkl_path']
save_path = args['save_path']
prediction_path = args['prediction_path']

path_tree_model = 'cifar10_vgg_tree_coreml.model'
select_model = 'cifar10_vgg_coreml_enhance_model_60.mlmodel'
path_graph_model = 'cifar10_vgg_coreml_node2vec.pkl'

X = np.array(pickle.load(open(pkl_path, 'rb')))
X = check_X(X)
X_pre = np.array(pickle.load(open(prediction_path, 'rb')))

x_statistic_feature = get_statistic_feature(X_pre)
x_graph_vec = get_graph_feature(X_pre, path_tree_model, path_graph_model)

print(X.shape)
print(X_pre.shape)
print(x_statistic_feature.shape)
print(x_graph_vec.shape)

def get_shape_1_32_32_3(x):
    x = x.reshape(1, 32, 32, 3)
    x = x.astype(np.float32)
    return x

def get_shape_1_10(x):
    x = x.reshape(1, 10)
    x = x.astype(np.float32)
    return x

def get_shape_1_4(x):
    x = x.reshape(1, 4)
    x = x.astype(np.float32)
    return x


def get_shape_1_128(x):
    x = x.reshape(1, 128)
    x = x.astype(np.float32)
    return x

model = coremltools.models.MLModel(select_model)
res_list = []
for i in range(len(X)):
    x0 = get_shape_1_32_32_3(X[i])
    x1 = get_shape_1_10(X_pre[i])
    x2 = get_shape_1_4(x_statistic_feature[i])
    x3 = get_shape_1_128(x_graph_vec[i])
    r = model.predict({'input_1': x0, 'input_2': x1, 'input_3': x2, 'input_4': x3})
    res_list.append(r)
    print('finished:', i)


pickle.dump(res_list, open(save_path, 'wb'), protocol=4)