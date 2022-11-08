import tensorflow as tf

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_file='resnet.h5', input_shapes={"input_1": [1, 64, 64, 3]})
converter.post_training_quantize = True

tflite_model = converter.convert()
open('resnet.tflite', "wb").write(tflite_model)
