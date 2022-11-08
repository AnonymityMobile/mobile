from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from PACE import PACE_selection
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import train_test_split
model = load_model('densenet.h5')


f = open('/home/users/pycharm/data/tiny_imagenet_pkl/tiny_imagenet_data.pkl', 'rb')
X = pickle.load(f)
f = open('/home/users/pycharm/data/tiny_imagenet_pkl/tiny_imagenet_label.pkl', 'rb')
y = pickle.load(f)
num_classes = 200
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

prediction_np = model.predict(x_test)

prediction_label_np = np.argmax(prediction_np, axis=1)
real_label_np = np.argmax(y_test, axis=1)
acc = accuracy_score(real_label_np, prediction_label_np)

index_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
dic = {}
for i in index_list:
    selected_index = PACE_selection(model, x_test, i)
    # selected_index = [731, 9999, 9998, 9997, 5481, 6680, 5268, 2770, 725, 3059]
    selected_y_test = y_test[selected_index]
    selected_pre_test = prediction_np[selected_index]

    prediction_label_np = np.argmax(selected_pre_test, axis=1)
    real_label_np = np.argmax(selected_y_test, axis=1)
    tmp_acc = accuracy_score(real_label_np, prediction_label_np)
    diff = abs(acc - tmp_acc)
    res_acc = pow(diff, 2)
    res = np.sqrt(res_acc)

    dic[i] = res

pickle.dump(dic, open('dic_pace_imagenet_densenet.pkl', 'wb'))


