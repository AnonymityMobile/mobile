import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.datasets import cifar10

num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


def get_shape_1_32_32_3(x):
    x = x.reshape(1, 32, 32, 3)
    x = x.astype(np.float32)
    return x


interpreter = tf.lite.Interpreter(model_path='cnn.tflite')
interpreter.allocate_tensors()
output = interpreter.get_output_details()[0]
input = interpreter.get_input_details()[0]


def model_prediction(x_test):
    predict_list = []

    for x in x_test:
        x = get_shape_1_32_32_3(x)
        interpreter.set_tensor(input['index'], x)
        interpreter.invoke()
        p_value = list(interpreter.get_tensor(output['index'])[0])
        predict_list.append(p_value)

    save_data = open('cifar10_cnn_prediction_train.pkl', 'wb')
    pickle.dump(predict_list, save_data)


if __name__ == '__main__':
    model_prediction(x_train)






