import pickle
import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

dic_mapping = {'n02110627': 253, 'n02088094': 161,  'n02088238': 162, 'n02093647': 182, 'n02090622': 170,
                'n02096585': 196, 'n02106382': 234, 'n02112137': 261, 'n02101556': 217, 'n02096437': 195, 'n02108915': 246,
                'n02109047': 247, 'n02105056': 225, 'n02102973': 222, 'n02090721': 171}

label_str_list = pickle.load(open('dog_image_label.pkl', 'rb'))
real_label_np = np.array([dic_mapping[i] for i in label_str_list])


def get_path_list(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.split('/')[-1].startswith('enhancement_prediction_dogscan') and file_absolute_path.endswith('.pkl') and 'enhancement_prediction_dogscanner' not in file_absolute_path:
                    path_list.append(file_absolute_path)
    return path_list


path_list = get_path_list('/mnt/irisgpfs/users/pycharm/data/dog_image_pkl')
key_list = [i.split('/')[-1] for i in path_list]

dic = {}
for i in range(len(path_list)):
    prediction_np = pickle.load(open(path_list[i], 'rb'))
    prediction_label_np = np.argmax(prediction_np, axis=1)
    acc = accuracy_score(real_label_np, prediction_label_np)
    dic[key_list[i]] = acc

print(len(dic))
print(dic)

