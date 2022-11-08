import tensorflow as tf
import pickle
import coremltools
import numpy as np
from sklearn.model_selection import train_test_split


def get_shape_1_64_64_3(x):
    x = x.reshape(1, 64, 64, 3)
    x = x.astype(np.float32)
    return x


f = open('tiny_imagenet_data.pkl', 'rb')
X = pickle.load(f)
f = open('tiny_imagenet_label.pkl', 'rb')
y = pickle.load(f)

num_classes = 200
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


def model_prediction(x_test, model, save_path):
    predict_list = []
    for x in x_test:
        x = get_shape_1_64_64_3(x)
        predict = list(model.predict({'input_1': get_shape_1_64_64_3(x)})['Identity'][0])
        predict_list.append(predict)
    predict_np = np.array(predict_list)
    save_data = open(save_path, 'wb')
    pickle.dump(predict_np, save_data)


model = coremltools.models.MLModel('resnet.mlmodel')
model_prediction(x_test, model, 'tinyimagenet_resnet_prediction_test_coreml.pkl')


