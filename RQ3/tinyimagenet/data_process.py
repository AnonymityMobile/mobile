import os
import numpy as np
import cv2
import pickle

def get_path_list(path_dir_compile):
    model_path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.JPEG'):
                    model_path_list.append(file_absolute_path)
    return model_path_list


path_list = get_path_list('imagenet/tiny-imagenet-200/train')
label_list = [i.split('/train/')[-1].split('/')[0].strip() for i in path_list]
label_set = list(set(label_list))
dic = dict(zip(label_set, range(len(label_set))))
label_np = np.array([dic[i] for i in label_list])

img_np = []
for i in path_list:
    img = cv2.imread(i)
    img_np.append(img)
img_np = np.array(img_np)
print(img_np.shape)
print(label_np.shape)


output = open('tiny_imagenet_data.pkl', 'wb')
pickle.dump(img_np, output)

output = open('tiny_imagenet_label.pkl', 'wb')
pickle.dump(label_np, output)
