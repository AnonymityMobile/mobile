import tensorflow as tf
import coremltools
from coremltools.models.neural_network import quantization_utils
from coremltools.models.neural_network.quantization_utils import AdvancedQuantizedLayerSelector

selector = AdvancedQuantizedLayerSelector(
    # skip_layer_types=['batchnorm', 'bias', 'depthwiseConv'],
    skip_layer_types=['batchnorm', 'depthwiseConv'],
    minimum_conv_kernel_channels=4,
    minimum_conv_weight_count=4096
)


def model_covert_coreml(h5_path, save_path, bit_num):

    model = tf.keras.models.load_model(h5_path)
    coreml_model = coremltools.convert(model)
    coreml_quan = quantization_utils.quantize_weights(coreml_model,
                                                      nbits=bit_num,
                                                      # quantization_mode="linear_symmetric",
                                                      selector=selector
                                                      )
    coreml_quan.save(save_path)


model_covert_coreml('resnet.h5', 'resnet.mlmodel', 8)
