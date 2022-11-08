import os
import tensorflow as tf
import numpy as np
import datetime
import cv2
import re
import pickle


model_path = 'identify_anything_search_by_image.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()


def get_shape_1_224_224_3(x_data):
    x = cv2.resize(x_data, (224, 224), interpolation=cv2.INTER_CUBIC)
    x = x.reshape(1, 224, 224, 3)
    return x


def get_shape_1_300_300_3(x_data):
    x = cv2.resize(x_data, (300, 300), interpolation=cv2.INTER_CUBIC)
    x = x.reshape(1, 300, 300, 3)
    return x


def get_model_prediction(pic):
    x = get_shape_1_224_224_3(pic)
    output = interpreter.get_output_details()[0]
    input = interpreter.get_input_details()[0]
    interpreter.set_tensor(input['index'], x)
    interpreter.invoke()
    p_value_list = list(interpreter.get_tensor(output['index'])[0])
    return p_value_list


def get_all_prediction(X_np):
    prediction_list = []
    for pic in X_np:
        prediction = get_model_prediction(pic)
        prediction_list.append(prediction)
    prediction_np = np.array(prediction_list)
    return prediction_np


if __name__ == '__main__':
    image_np = pickle.load(open('anything_image_data.pkl', 'rb'))
    prediction_np = get_all_prediction(image_np)
    save_data = open('anything_prediction_test.pkl', 'wb')
    pickle.dump(prediction_np, save_data)

