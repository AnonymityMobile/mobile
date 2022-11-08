import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

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


def get_shape_1_32_32_3(x):
    x = x.reshape(1, 32, 32, 3)
    x = x.astype(np.float32)
    return x


def get_shape_1_28_28_1(x):
    x = x.reshape(1, 28, 28, 1)
    x = x.astype(np.float32)
    return x


def get_shape_1_64_64_3(x):
    x = x.reshape(1, 64, 64, 3)
    x = x.astype(np.float32)
    return x


interpreter = tf.lite.Interpreter(model_path='densenet.tflite')
interpreter.allocate_tensors()
output = interpreter.get_output_details()[0]
input = interpreter.get_input_details()[0]


def model_prediction(x_test):
    predict_list = []

    for x in x_test:
        x = get_shape_1_64_64_3(x)
        interpreter.set_tensor(input['index'], x)
        interpreter.invoke()
        p_value = list(interpreter.get_tensor(output['index'])[0])
        predict_list.append(p_value)

    save_data = open('tinyimagenet_densenet_prediction_train.pkl', 'wb')
    pickle.dump(predict_list, save_data)


if __name__ == '__main__':
    model_prediction(x_train)



