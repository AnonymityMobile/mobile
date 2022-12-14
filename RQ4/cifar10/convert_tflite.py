import tensorflow as tf

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_file='cifar10_vgg_tflite_enhance_model_60.h5')
converter.post_training_quantize = True

tflite_model = converter.convert()
open('cifar10_vgg_tflite_enhance_model_60.tflite', "wb").write(tflite_model)

