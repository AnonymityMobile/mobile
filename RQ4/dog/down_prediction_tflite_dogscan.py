import tensorflow as tf
import numpy as np
import argparse
import pickle
from tensorflow.keras.datasets import cifar10
import joblib
import cv2

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


path_tree_model = 'dogscan_tree.model'
select_model = 'dogscan_enhance_model_60.tflite'
path_graph_model = 'dogscan_node2vec.pkl'

X = np.array(pickle.load(open(pkl_path, 'rb')))
X = check_X(X)
X_pre = np.array(pickle.load(open(prediction_path, 'rb')))

x_statistic_feature = get_statistic_feature(X_pre)
x_graph_vec = get_graph_feature(X_pre, path_tree_model, path_graph_model)

interpreter = tf.lite.Interpreter(model_path=select_model)
interpreter.allocate_tensors()
output = interpreter.get_output_details()[0]
input = interpreter.get_input_details()


def get_shape_1_1001(x):
    x = x.reshape(1, 1001)
    x = x.astype(np.float32)
    return x


def get_shape_1_128(x):
    x = x.reshape(1, 128)
    x = x.astype(np.float32)
    return x


def get_shape_1_127(x):
    x = x.reshape(1, 127)
    x = x.astype(np.float32)
    return x


def get_shape_1_4(x):
    x = x.reshape(1, 4)
    x = x.astype(np.float32)
    return x


def get_shape_1_150_150_3(x):
    x = x.reshape(1, 150, 150, 3)
    x = x.astype(np.float32)
    return x


def model_prediction(x_prediction, x_statistic_feature, x_graph_vec, x_pic):
    print('========start_prediction=======')
    prediction_list = []
    for i in range(len(x_prediction)):
        x0 = get_shape_1_1001(x_prediction[i])
        x1 = get_shape_1_4(x_statistic_feature[i])
        x2 = get_shape_1_128(x_graph_vec[i])
        x3 = get_shape_1_150_150_3(x_pic[i])

        interpreter.set_tensor(input[0]['index'], x0)
        interpreter.set_tensor(input[1]['index'], x1)
        interpreter.set_tensor(input[2]['index'], x2)
        interpreter.set_tensor(input[3]['index'], x3)
        interpreter.invoke()
        p_value = list(interpreter.get_tensor(output['index'])[0])
        prediction_list.append(p_value)
    save_data = open(save_path, 'wb')
    pickle.dump(np.array(prediction_list), save_data)



if __name__ == '__main__':
    model_prediction(X_pre, x_statistic_feature, x_graph_vec, X)




