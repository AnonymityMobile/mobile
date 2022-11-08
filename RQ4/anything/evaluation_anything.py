import pickle
import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

dic_mapping = {'n01872401': 103, 'n01806143': 85, 'n04254680': 806, 'n04254777': 807, 'n04355933': 837,
              'n04371774': 844, 'n04483307': 872, 'n04507155': 880, 'n04532670': 889, 'n04552348': 896}

label_str_list = pickle.load(open('anything_image_label.pkl', 'rb'))
real_label_np = np.array([dic_mapping[i] for i in label_str_list])


def get_path_list(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.split('/')[-1].startswith('enhancement_prediction_anything') and file_absolute_path.endswith('.pkl'):
                    path_list.append(file_absolute_path)
    return path_list


path_list = get_path_list('/mnt/irisgpfs/users/pycharm/data/anything_image_pkl')
key_list = [i.split('/')[-1] for i in path_list]

dic = {}
for i in range(len(path_list)):
    prediction_np = pickle.load(open(path_list[i], 'rb'))
    prediction_label_np = np.argmax(prediction_np, axis=1)
    acc = accuracy_score(real_label_np, prediction_label_np)
    dic[key_list[i]] = acc

print(len(dic))
print(dic)

