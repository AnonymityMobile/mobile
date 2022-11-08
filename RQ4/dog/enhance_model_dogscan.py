from __future__ import print_function
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os
import pickle
import cv2
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from config_path import *
import argparse


dic_mapping = {'n02110627': 253, 'n02088094': 161,  'n02088238': 162, 'n02093647': 182, 'n02090622': 170,
                'n02096585': 196, 'n02106382': 234, 'n02112137': 261, 'n02101556': 217, 'n02096437': 195, 'n02108915': 246,
                'n02109047': 247, 'n02105056': 225, 'n02102973': 222, 'n02090721': 171}
n_vec=1001


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


def enhance_model(ai_model_output_dimension, statistic_output_dimension, graph_shape_output_dimension, x_train):
    input = Input(shape=x_train.shape[1:], dtype='float32')
    x = Conv2D(32, (3, 3), padding='same')(input)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    interlayer_model_out = Dense(ai_model_output_dimension)(x)

    ai_model_output = Input(shape=(ai_model_output_dimension,))
    statistic_output = Input(shape=(statistic_output_dimension,))
    graph_shape_output = Input(shape=(graph_shape_output_dimension,))

    x = tf.keras.layers.concatenate([ai_model_output, statistic_output, graph_shape_output, interlayer_model_out])
    x = Dense(ai_model_output_dimension)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=[ai_model_output, statistic_output, graph_shape_output, input], outputs=x)

    sgd = SGD(lr=0.01, decay=0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def get_y_vec(dic_mapping, n_vec, y):
    y_int = [dic_mapping[i] for i in y]
    y_vec = []
    for i in y_int:
        tmp_vec = [0]*n_vec
        tmp_vec[i] = 1
        y_vec.append(tmp_vec)
    y_vec = np.array(y_vec)
    return y_vec


def get_X_ypre(config_path_list, path_y):

    y = pickle.load(open(path_y, 'rb'))

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

        y_train = get_y_vec(dic_mapping, n_vec, y_train)
        y_test = get_y_vec(dic_mapping, n_vec, y_test)

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


save_flag = 'dogscan'
path_tree_model = 'dogscan_tree.model'
path_graph_model = 'dogscan_node2vec.pkl'

x_train, x_train_prediction, y_train, x_test, x_test_prediction, y_test = get_X_ypre(path_pre_data_dogscan, 'dog_image_label.pkl')

print(x_train.shape)
print(x_train_prediction.shape)
print(y_train.shape)
print(x_test.shape)
print(x_test_prediction.shape)
print(y_test.shape)

ai_model_output_dimension = 1001
statistic_output_dimension = 4
graph_shape_output_dimension = 128

x_train_statistic_feature = get_statistic_feature(x_train_prediction)
x_train_graph_vec = get_graph_feature(x_train_prediction, path_tree_model, path_graph_model)

x_text_statistic_feature = get_statistic_feature(x_test_prediction)
x_test_graph_vec = get_graph_feature(x_test_prediction, path_tree_model, path_graph_model)

model = enhance_model(ai_model_output_dimension, statistic_output_dimension, graph_shape_output_dimension, x_train)

model.fit([x_train_prediction, x_train_statistic_feature, x_train_graph_vec, x_train], y_train,
          validation_data=([x_test_prediction, x_text_statistic_feature, x_test_graph_vec, x_test], y_test),
          shuffle=True, epochs=10, batch_size=64)
model.save(save_flag+'_'+'enhance_model_10.h5')
model.fit([x_train_prediction, x_train_statistic_feature, x_train_graph_vec, x_train], y_train,
          validation_data=([x_test_prediction, x_text_statistic_feature, x_test_graph_vec, x_test], y_test),
          shuffle=True, epochs=10, batch_size=64)
model.save(save_flag+'_'+'enhance_model_20.h5')
model.fit([x_train_prediction, x_train_statistic_feature, x_train_graph_vec, x_train], y_train,
          validation_data=([x_test_prediction, x_text_statistic_feature, x_test_graph_vec, x_test], y_test),
          shuffle=True, epochs=10, batch_size=64)
model.save(save_flag+'_'+'enhance_model_30.h5')
model.fit([x_train_prediction, x_train_statistic_feature, x_train_graph_vec, x_train], y_train,
          validation_data=([x_test_prediction, x_text_statistic_feature, x_test_graph_vec, x_test], y_test),
          shuffle=True, epochs=10, batch_size=64)
model.save(save_flag+'_'+'enhance_model_40.h5')
model.fit([x_train_prediction, x_train_statistic_feature, x_train_graph_vec, x_train], y_train,
          validation_data=([x_test_prediction, x_text_statistic_feature, x_test_graph_vec, x_test], y_test),
          shuffle=True, epochs=10, batch_size=64)
model.save(save_flag+'_'+'enhance_model_50.h5')
model.fit([x_train_prediction, x_train_statistic_feature, x_train_graph_vec, x_train], y_train,
          validation_data=([x_test_prediction, x_text_statistic_feature, x_test_graph_vec, x_test], y_test),
          shuffle=True, epochs=10, batch_size=64)
model.save(save_flag+'_'+'enhance_model_60.h5')
model.fit([x_train_prediction, x_train_statistic_feature, x_train_graph_vec, x_train], y_train,
          validation_data=([x_test_prediction, x_text_statistic_feature, x_test_graph_vec, x_test], y_test),
          shuffle=True, epochs=10, batch_size=64)
model.save(save_flag+'_'+'enhance_model_70.h5')
model.fit([x_train_prediction, x_train_statistic_feature, x_train_graph_vec, x_train], y_train,
          validation_data=([x_test_prediction, x_text_statistic_feature, x_test_graph_vec, x_test], y_test),
          shuffle=True, epochs=10, batch_size=64)
model.save(save_flag+'_'+'enhance_model_80.h5')
