import tensorflow as tf

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_file='vgg.h5')
converter.post_training_quantize = True

tflite_model = converter.convert()
open('vgg.tflite', "wb").write(tflite_model)

