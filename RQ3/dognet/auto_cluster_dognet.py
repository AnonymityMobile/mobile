import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import random
import pickle
import matplotlib
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

dic_mapping = {'n02110627': 0, 'n02088094': 1,  'n02088238': 9, 'n02093647': 11, 'n02090622': 20,
               'n02096585': 21, 'n02106382': 22, 'n02112137': 34, 'n02101556': 35, 'n02096437': 39, 'n02108915': 50,
               'n02109047': 56, 'n02105056': 59, 'n02102973': 63, 'n02090721': 64}
n_vec=127

def Margin_score(x):
    output_sort = np.sort(x)
    margin_score = output_sort[:, -1] - output_sort[:, -2]
    return margin_score


def DeepGini_score(x):
    gini_score = 1 - np.sum(np.power(x, 2), axis=1)
    return gini_score


def Entropy_score(x):
    score = -np.sum(x * np.log(x), axis=1)
    return score


def LeastConfidence_score(x):
    max_pre = x.max(1)
    return max_pre


def Variance_score(x):
    var = np.var(x, axis=1)
    return var


def get_encoder_feature(model, X):
    get_encoded = K.function([model.layers[0].input], [model.layers[4].output])
    X_encoded = get_encoded([X])[0]
    return X_encoded


def get_uncertainty_feature(prediction):
    df = pd.DataFrame(columns=['margin', 'deepgini', 'entropy', 'least', 'variance'])
    df['margin'] = Margin_score(prediction)
    df['deepgini'] = DeepGini_score(prediction)
    df['entropy'] = Entropy_score(prediction)
    df['least'] = LeastConfidence_score(prediction)
    df['variance'] = Variance_score(prediction)
    df['idx'] = list(range(len(prediction)))
    return df


def get_number_cluster(X, max_n):
    res = 0
    for i in range(2, max_n):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
        cluster_labels = kmeans.labels_
        sc = silhouette_score(X, cluster_labels)
        print('i=', i, 'sc=', sc)
        if sc < res:
            return i-1
        else:
            res = sc
    return i


def get_cluster_label(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    cluster_labels = list(kmeans.labels_)
    return cluster_labels


def get_select_idx(score_np, groups_n, select_n):
    group_number = int(len(score_np) / groups_n)
    select_idx_list = list(np.argsort(score_np))
    res_select_idx = []
    left = 0
    for group in range(groups_n):
        tmp_group = select_idx_list[left: left+group_number]
        res_select_idx += random.sample(tmp_group, select_n)
        left = left+group_number
    return res_select_idx


def get_pre_result(iter_n, max_select_n_group, df, metric_type):
    res = []
    for i in range(1, max_select_n_group):
        res_acc = 0
        for _ in range(iter_n):
            res_select_idx = []
            for cluster_label in list(set(df['cluster'])):
                tmp_df = df[df['cluster'] == cluster_label]
                dic_mapping = dict(zip(list(range(len(tmp_df))), tmp_df['idx']))
                tmp_res_select_idx = get_select_idx(tmp_df[metric_type].to_numpy(), 5, i)
                tmp_mapping_select_idx = [dic_mapping[i] for i in tmp_res_select_idx]
                res_select_idx += tmp_mapping_select_idx

            y_pre = df['pre'].to_numpy()[res_select_idx]
            y_real = df['real'].to_numpy()[res_select_idx]
            tmp_acc = accuracy_score(y_real, y_pre)
            diff = abs(acc - tmp_acc)
            res_acc += pow(diff, 2)
        res_acc = np.sqrt(res_acc * 1.0 / iter_n)
        res.append(res_acc)
    return res


def get_random_result(iter_n, df):
    res = []
    for i in range(1, 16):
        res_acc = 0
        for _ in range(iter_n):
            res_select_idx = random.sample(list(range(len(df))), i*10)
            y_pre = df['pre'].to_numpy()[res_select_idx]
            y_real = df['real'].to_numpy()[res_select_idx]
            tmp_acc = accuracy_score(y_real, y_pre)
            diff = abs(acc - tmp_acc)
            res_acc += pow(diff, 2)
        res_acc = np.sqrt(res_acc * 1.0 / iter_n)
        res.append(res_acc)
    return res


def get_y_vec(dic_mapping, n_vec, y):
    y_int = [dic_mapping[i] for i in y]
    y_vec = []
    for i in y_int:
        tmp_vec = [0]*n_vec
        tmp_vec[i] = 1
        y_vec.append(tmp_vec)
    y_vec = np.array(y_vec)
    return y_vec


f = open('dognet_prediction_test.pkl', 'rb')
prediction_np = np.array(pickle.load(f))

f = open('dog_image_data.pkl', 'rb')
X = pickle.load(f)
f = open('dog_image_label.pkl', 'rb')
y = pickle.load(f)
y = get_y_vec(dic_mapping, n_vec, y)

X = X.astype('float32') / 255
X = X.reshape((len(X), len(X[0].flatten())))


model_encoder = load_model('auto_encoder_dog.h5')
X_encoded_feature = get_encoder_feature(model_encoder, X)
print(X_encoded_feature.shape)

prediction_label_np = np.argmax(prediction_np, axis=1)
real_label_np = np.argmax(y, axis=1)
acc = accuracy_score(real_label_np, prediction_label_np)

print('acc=', acc)

# df = get_uncertainty_feature(prediction_np)
# n_clusters = get_number_cluster(X_encoded_feature, 10)
# cluster_labels = get_cluster_label(X_encoded_feature, n_clusters)
# df['cluster'] = cluster_labels
# df['pre'] = prediction_label_np
# df['real'] = real_label_np
#
# print(df.head())
# print('=====')
#
# margin = get_pre_result(100, 16, df, 'margin')
# deepgini = get_pre_result(100, 16, df, 'deepgini')
# entropy = get_pre_result(100, 16, df, 'entropy')
# least = get_pre_result(100, 16, df, 'least')
# variance = get_pre_result(100, 16, df, 'variance')
# random_re = get_random_result(100, df)
#
#
# index_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
# plt.style.use('ggplot')
# fig1, ax1 = plt.subplots(figsize=(15, 12))
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
# plt.plot(index_list, margin, marker="o", markersize=5, color='blue')
# plt.plot(index_list, deepgini, marker="o", markersize=5, color='darkgrey')
# plt.plot(index_list, entropy, marker="o", markersize=5, color='rebeccapurple')
# plt.plot(index_list, least, marker="o", markersize=5, color='lightsalmon')
# plt.plot(index_list, variance, marker="o", markersize=5, color='yellow')
# plt.plot(index_list, random_re, marker="o", markersize=5, color='black')
# plt.legend(['EDCM', 'EDCD', 'EDCE', 'EDCL', 'EDCV', 'Random'], frameon=True, prop={'size': 30}, loc=1)
# plt.xlabel('Number of selected test inputs', color='black', size=30)
# plt.ylabel('MSE ', color='black', size=30)
#
# plt.yticks(size=30, color='black')
# plt.xticks(size=30, color='black')
# fig1.savefig('../pictures/dognet_tflite.pdf')
# plt.show()
#

