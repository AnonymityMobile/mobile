from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
from CES import CES_selection
import numpy as np
from sklearn.metrics import accuracy_score
import pickle

model = load_model('cnn.h5')

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

prediction_np = model.predict(x_test)

prediction_label_np = np.argmax(prediction_np, axis=1)
real_label_np = np.argmax(y_test, axis=1)
acc = accuracy_score(real_label_np, prediction_label_np)

index_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
dic = {}
for i in index_list:
    selected_index = CES_selection(model, x_test, i)
    selected_y_test = y_test[selected_index]
    selected_pre_test = prediction_np[selected_index]
    prediction_label_np = np.argmax(selected_pre_test, axis=1)
    real_label_np = np.argmax(selected_y_test, axis=1)
    tmp_acc = accuracy_score(real_label_np, prediction_label_np)
    diff = abs(acc - tmp_acc)
    res_acc = pow(diff, 2)
    res = np.sqrt(res_acc)

    dic[i] = res

print(dic)



