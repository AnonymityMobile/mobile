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


path_list = get_path_list('dog/base')
label_np = np.array([i.split('/')[-1].split('_')[0].strip() for i in path_list])


img_np = []
for i in path_list:
    img = cv2.imread(i)
    img_np.append(img)
img_np = np.array(img_np)
print(img_np.shape)


output = open('dog_image_data.pkl', 'wb')
pickle.dump(img_np, output)

output = open('dog_image_label.pkl', 'wb')
pickle.dump(label_np, output)
