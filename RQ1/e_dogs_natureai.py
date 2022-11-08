import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from config import *
from basic_transform import *

# modify
dic_mapping=dic_e_dogs_natureai_lite
path_dir_compile=data_path_8
file_name = save_path_8
x = 299
model_path = model_path_8


interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()


def get_acc(label_idx_list, prediction_idx_list):
    n = 0
    for i in range(len(label_idx_list)):
        if label_idx_list[i] == prediction_idx_list[i]:
            n += 1
    acc = n*1.0 / len(label_idx_list)
    return acc

def check_x(pic):
    if np.max(pic)>5:
        return pic / 255.0
    return pic / 1.0


def get_shape_x_x_3(x_data, x):
    img = cv2.resize(x_data, (x, x), interpolation=cv2.INTER_CUBIC)
    return img


def get_shape_1_x_x_3(x_data, x):
    img = cv2.resize(x_data, (x, x), interpolation=cv2.INTER_CUBIC)
    img = img.reshape(1, x, x, 3)
    return img


def get_model_prediction(pic, x):
    pic = get_shape_1_x_x_3(pic, x)
    pic = check_x(pic)
    pic = np.array(pic, dtype=np.float32)
    output = interpreter.get_output_details()[0]
    input = interpreter.get_input_details()[0]

    interpreter.set_tensor(input['index'], pic)
    interpreter.invoke()
    p_value = list(interpreter.get_tensor(output['index'])[0])
    return p_value


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


def input_output_detail(path_model):
    interpreter = tf.lite.Interpreter(model_path=path_model)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    output_shape = output_details[0]['shape']
    print(input_shape)
    print(output_shape)

# input_output_detail('dogpedia.dogpedia.tflite')


def get_path_label(dic_mapping, path_dir_compile):
    path_list = []
    res_path_list = []
    res_idx_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.JPEG'):
                    path_list.append(file_absolute_path)
    for i in path_list:
        for j in dic_mapping:
            if j in i:
                res_path_list.append(i)
                res_idx_list.append(dic_mapping[j])
                break
    res_dic = dict(zip(res_path_list, res_idx_list))
    return res_dic


def read_image(path_dic):
    image_path_list = list(path_dic.keys())
    label_idx_list = list(path_dic.values())
    image_list = []
    for path in image_path_list:
        pic = cv2.imread(path)
        pic = get_shape_x_x_3(pic, x)
        image_list.append(pic)
    image_np = np.array(image_list)
    return image_np, label_idx_list


path_dic = get_path_label(dic_mapping=dic_mapping, path_dir_compile=path_dir_compile)
n_pic = len(path_dic)
write_result('number of pics:'+str(n_pic), file_name)


image_np, label_idx_list = read_image(path_dic)
base_np = image_np.copy()
image_augmentation_width_shift = run(image_np.copy(), 'augmentation_width_shift')
image_augmentation_height_shift = run(image_np.copy(), 'augmentation_height_shift')
image_augmentation_horizontal_flip = run(image_np.copy(), 'augmentation_horizontal_flip')
image_augmentation_vertical_flip = run(image_np.copy(), 'augmentation_vertical_flip')
image_augmentation_rotation = run(image_np.copy(), 'augmentation_rotation')
image_augmentation_brightness = run(image_np.copy(), 'augmentation_brightness')
image_augmentation_zoom = run(image_np.copy(), 'augmentation_zoom')
image_augmentation_featurewise_std_normalization = run(image_np.copy(), 'augmentation_featurewise_std_normalization')
image_augmentation_zca_whitening = run(image_np.copy(), 'augmentation_zca_whitening')
image_augmentation_shear_range = run(image_np.copy(), 'augmentation_shear_range')
image_augmentation_channel_shift_range = run(image_np.copy(), 'augmentation_channel_shift_range')
image_noise_random = run(image_np.copy(), 'noise_random')
image_noise_salt_pepper = run(image_np.copy(), 'noise_salt_pepper')
image_noise_gasuss = run(image_np.copy(), 'noise_gasuss')
image_contrast = run(image_np.copy(), 'contrast')


def get_all_acc(image_np):
    pre_list = []
    for image in image_np:
        pre = get_model_prediction(image, x)
        pre_list.append(pre)
    prediction_idx_list = [i.index(max(i)) for i in pre_list]
    acc = get_acc(label_idx_list, prediction_idx_list)
    return acc


dic_res = {}
dic_res['image_np'] = get_all_acc(base_np)
dic_res['image_augmentation_width_shift'] = get_all_acc(image_augmentation_width_shift)
dic_res['image_augmentation_height_shift'] = get_all_acc(image_augmentation_height_shift)
dic_res['image_augmentation_horizontal_flip'] = get_all_acc(image_augmentation_horizontal_flip)
dic_res['image_augmentation_vertical_flip'] = get_all_acc(image_augmentation_vertical_flip)
dic_res['image_augmentation_rotation'] = get_all_acc(image_augmentation_rotation)
dic_res['image_augmentation_brightness'] = get_all_acc(image_augmentation_brightness)
dic_res['image_augmentation_zoom'] = get_all_acc(image_augmentation_zoom)
dic_res['image_augmentation_featurewise_std_normalization'] = get_all_acc(image_augmentation_featurewise_std_normalization)
dic_res['image_augmentation_zca_whitening'] = get_all_acc(image_augmentation_zca_whitening)
dic_res['image_augmentation_shear_range'] = get_all_acc(image_augmentation_shear_range)
dic_res['image_augmentation_channel_shift_range'] = get_all_acc(image_augmentation_channel_shift_range)
dic_res['image_noise_random'] = get_all_acc(image_noise_random)
dic_res['image_noise_salt_pepper'] = get_all_acc(image_noise_salt_pepper)
dic_res['image_noise_gasuss'] = get_all_acc(image_noise_gasuss)
dic_res['image_contrast'] = get_all_acc(image_contrast)

write_result(str(dic_res), file_name)
print(dic_res)





