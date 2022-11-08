num_classes = 200
tiny_imagenet_data = 'tiny_imagenet_data.pkl'
tiny_imagenet_label = 'tiny_imagenet_label.pkl'

path_pre_data_densenet_tflite_data = [
    ['augmentation_brightness_tiny_imagenet_data.pkl', 'prediction_augmentation_brightness_tiny_imagenet_data_densenet_tflite.pkl'],
    ['augmentation_channel_shift_range_tiny_imagenet_data.pkl', 'prediction_augmentation_channel_shift_range_tiny_imagenet_data_densenet_tflite.pkl'],
    ['augmentation_featurewise_std_normalization_tiny_imagenet_data.pkl', 'prediction_augmentation_featurewise_std_normalization_tiny_imagenet_data_densenet_tflite.pkl'],
    ['augmentation_height_shift_tiny_imagenet_data.pkl', 'prediction_augmentation_height_shift_tiny_imagenet_data_densenet_tflite.pkl'],
    ['augmentation_horizontal_flip_tiny_imagenet_data.pkl', 'prediction_augmentation_horizontal_flip_tiny_imagenet_data_densenet_tflite.pkl'],
    ['augmentation_rotation_tiny_imagenet_data.pkl', 'prediction_augmentation_rotation_tiny_imagenet_data_densenet_tflite.pkl'],
    ['augmentation_shear_range_tiny_imagenet_data.pkl', 'prediction_augmentation_shear_range_tiny_imagenet_data_densenet_tflite.pkl'],
    ['augmentation_vertical_flip_tiny_imagenet_data.pkl', 'prediction_augmentation_vertical_flip_tiny_imagenet_data_densenet_tflite.pkl'],
    ['augmentation_width_shift_tiny_imagenet_data.pkl', 'prediction_augmentation_width_shift_tiny_imagenet_data_densenet_tflite.pkl'],
    ['augmentation_zca_whitening_tiny_imagenet_data.pkl', 'prediction_augmentation_zca_whitening_tiny_imagenet_data_densenet_tflite.pkl'],
    ['augmentation_zoom_tiny_imagenet_data.pkl', 'prediction_augmentation_zoom_tiny_imagenet_data_densenet_tflite.pkl'],
    ['contrast_tiny_imagenet_data.pkl', 'prediction_contrast_tiny_imagenet_data_densenet_tflite.pkl'],
    ['noise_gasuss_tiny_imagenet_data.pkl', 'prediction_noise_gasuss_tiny_imagenet_data_densenet_tflite.pkl'],
    ['noise_random_tiny_imagenet_data.pkl', 'prediction_noise_random_tiny_imagenet_data_densenet_tflite.pkl'],
    ['noise_salt_pepper_tiny_imagenet_data.pkl', 'prediction_noise_salt_pepper_tiny_imagenet_data_densenet_tflite.pkl'],
    ['tiny_imagenet_data.pkl', 'prediction_tiny_imagenet_data_densenet_tflite.pkl']
]


path_pre_data_resnet_tflite_data = [
    ['augmentation_brightness_tiny_imagenet_data.pkl', 'prediction_augmentation_brightness_tiny_imagenet_data_resnet_tflite.pkl'],
    ['augmentation_channel_shift_range_tiny_imagenet_data.pkl', 'prediction_augmentation_channel_shift_range_tiny_imagenet_data_resnet_tflite.pkl'],
    ['augmentation_featurewise_std_normalization_tiny_imagenet_data.pkl', 'prediction_augmentation_featurewise_std_normalization_tiny_imagenet_data_resnet_tflite.pkl'],
    ['augmentation_height_shift_tiny_imagenet_data.pkl', 'prediction_augmentation_height_shift_tiny_imagenet_data_resnet_tflite.pkl'],
    ['augmentation_horizontal_flip_tiny_imagenet_data.pkl', 'prediction_augmentation_horizontal_flip_tiny_imagenet_data_resnet_tflite.pkl'],
    ['augmentation_rotation_tiny_imagenet_data.pkl', 'prediction_augmentation_rotation_tiny_imagenet_data_resnet_tflite.pkl'],
    ['augmentation_shear_range_tiny_imagenet_data.pkl', 'prediction_augmentation_shear_range_tiny_imagenet_data_resnet_tflite.pkl'],
    ['augmentation_vertical_flip_tiny_imagenet_data.pkl', 'prediction_augmentation_vertical_flip_tiny_imagenet_data_resnet_tflite.pkl'],
    ['augmentation_width_shift_tiny_imagenet_data.pkl', 'prediction_augmentation_width_shift_tiny_imagenet_data_resnet_tflite.pkl'],
    ['augmentation_zca_whitening_tiny_imagenet_data.pkl', 'prediction_augmentation_zca_whitening_tiny_imagenet_data_resnet_tflite.pkl'],
    ['augmentation_zoom_tiny_imagenet_data.pkl', 'prediction_augmentation_zoom_tiny_imagenet_data_resnet_tflite.pkl'],
    ['contrast_tiny_imagenet_data.pkl', 'prediction_contrast_tiny_imagenet_data_resnet_tflite.pkl'],
    ['noise_gasuss_tiny_imagenet_data.pkl', 'prediction_noise_gasuss_tiny_imagenet_data_resnet_tflite.pkl'],
    ['noise_random_tiny_imagenet_data.pkl', 'prediction_noise_random_tiny_imagenet_data_resnet_tflite.pkl'],
    ['noise_salt_pepper_tiny_imagenet_data.pkl', 'prediction_noise_salt_pepper_tiny_imagenet_data_resnet_tflite.pkl'],
    ['tiny_imagenet_data.pkl', 'prediction_tiny_imagenet_data_resnet_tflite.pkl']
]


path_densenet_coreml = 'tinyimagenet_prediction_densenet_coreml/'
path_pre_data_densenet_coreml_data = [
    ['augmentation_brightness_tiny_imagenet_data.pkl', path_densenet_coreml + 'prediction_augmentation_brightness_tiny_imagenet_data_densenet_coreml.pkl'],
    ['augmentation_channel_shift_range_tiny_imagenet_data.pkl', path_densenet_coreml + 'prediction_augmentation_channel_shift_range_tiny_imagenet_data_densenet_coreml.pkl'],
    ['augmentation_featurewise_std_normalization_tiny_imagenet_data.pkl', path_densenet_coreml + 'prediction_augmentation_featurewise_std_normalization_tiny_imagenet_data_densenet_coreml.pkl'],
    ['augmentation_height_shift_tiny_imagenet_data.pkl', path_densenet_coreml + 'prediction_augmentation_height_shift_tiny_imagenet_data_densenet_coreml.pkl'],
    ['augmentation_horizontal_flip_tiny_imagenet_data.pkl', path_densenet_coreml + 'prediction_augmentation_horizontal_flip_tiny_imagenet_data_densenet_coreml.pkl'],
    ['augmentation_rotation_tiny_imagenet_data.pkl', path_densenet_coreml + 'prediction_augmentation_rotation_tiny_imagenet_data_densenet_coreml.pkl'],
    ['augmentation_shear_range_tiny_imagenet_data.pkl', path_densenet_coreml + 'prediction_augmentation_shear_range_tiny_imagenet_data_densenet_coreml.pkl'],
    ['augmentation_vertical_flip_tiny_imagenet_data.pkl', path_densenet_coreml + 'prediction_augmentation_vertical_flip_tiny_imagenet_data_densenet_coreml.pkl'],
    ['augmentation_width_shift_tiny_imagenet_data.pkl', path_densenet_coreml + 'prediction_augmentation_width_shift_tiny_imagenet_data_densenet_coreml.pkl'],
    ['augmentation_zca_whitening_tiny_imagenet_data.pkl', path_densenet_coreml + 'prediction_augmentation_zca_whitening_tiny_imagenet_data_densenet_coreml.pkl'],
    ['augmentation_zoom_tiny_imagenet_data.pkl', path_densenet_coreml + 'prediction_augmentation_zoom_tiny_imagenet_data_densenet_coreml.pkl'],
    ['contrast_tiny_imagenet_data.pkl', path_densenet_coreml + 'prediction_contrast_tiny_imagenet_data_densenet_coreml.pkl'],
    ['noise_gasuss_tiny_imagenet_data.pkl', path_densenet_coreml + 'prediction_noise_gasuss_tiny_imagenet_data_densenet_coreml.pkl'],
    ['noise_random_tiny_imagenet_data.pkl', path_densenet_coreml + 'prediction_noise_random_tiny_imagenet_data_densenet_coreml.pkl'],
    ['noise_salt_pepper_tiny_imagenet_data.pkl', path_densenet_coreml + 'prediction_noise_salt_pepper_tiny_imagenet_data_densenet_coreml.pkl']
]
tinyimagenet_densenet_prediction_test_coreml = path_densenet_coreml + 'tinyimagenet_densenet_prediction_test_coreml.pkl'
tinyimagenet_densenet_prediction_train_coreml = path_densenet_coreml + 'tinyimagenet_densenet_prediction_train_coreml.pkl'



path_resnet_coreml = 'tinyimagenet_prediction_resnet_coreml/'
path_pre_data_resnet_coreml_data = [
    ['augmentation_brightness_tiny_imagenet_data.pkl', path_resnet_coreml + 'prediction_augmentation_brightness_tiny_imagenet_data_resnet_coreml.pkl'],
    ['augmentation_channel_shift_range_tiny_imagenet_data.pkl',  path_resnet_coreml + 'prediction_augmentation_channel_shift_range_tiny_imagenet_data_resnet_coreml.pkl'],
    ['augmentation_featurewise_std_normalization_tiny_imagenet_data.pkl',  path_resnet_coreml + 'prediction_augmentation_featurewise_std_normalization_tiny_imagenet_data_resnet_coreml.pkl'],
    ['augmentation_height_shift_tiny_imagenet_data.pkl',  path_resnet_coreml + 'prediction_augmentation_height_shift_tiny_imagenet_data_resnet_coreml.pkl'],
    ['augmentation_horizontal_flip_tiny_imagenet_data.pkl',  path_resnet_coreml + 'prediction_augmentation_horizontal_flip_tiny_imagenet_data_resnet_coreml.pkl'],
    ['augmentation_rotation_tiny_imagenet_data.pkl',  path_resnet_coreml + 'prediction_augmentation_rotation_tiny_imagenet_data_resnet_coreml.pkl'],
    ['augmentation_shear_range_tiny_imagenet_data.pkl',  path_resnet_coreml + 'prediction_augmentation_shear_range_tiny_imagenet_data_resnet_coreml.pkl'],
    ['augmentation_vertical_flip_tiny_imagenet_data.pkl',  path_resnet_coreml + 'prediction_augmentation_vertical_flip_tiny_imagenet_data_resnet_coreml.pkl'],
    ['augmentation_width_shift_tiny_imagenet_data.pkl',  path_resnet_coreml + 'prediction_augmentation_width_shift_tiny_imagenet_data_resnet_coreml.pkl'],
    ['augmentation_zca_whitening_tiny_imagenet_data.pkl',  path_resnet_coreml + 'prediction_augmentation_zca_whitening_tiny_imagenet_data_resnet_coreml.pkl'],
    ['augmentation_zoom_tiny_imagenet_data.pkl',  path_resnet_coreml + 'prediction_augmentation_zoom_tiny_imagenet_data_resnet_coreml.pkl'],
    ['contrast_tiny_imagenet_data.pkl',  path_resnet_coreml + 'prediction_contrast_tiny_imagenet_data_resnet_coreml.pkl'],
    ['noise_gasuss_tiny_imagenet_data.pkl',  path_resnet_coreml + 'prediction_noise_gasuss_tiny_imagenet_data_resnet_coreml.pkl'],
    ['noise_random_tiny_imagenet_data.pkl',  path_resnet_coreml + 'prediction_noise_random_tiny_imagenet_data_resnet_coreml.pkl'],
    ['noise_salt_pepper_tiny_imagenet_data.pkl',  path_resnet_coreml + 'prediction_noise_salt_pepper_tiny_imagenet_data_resnet_coreml.pkl']
]

tinyimagenet_resnet_prediction_test_coreml = 'tinyimagenet_resnet_prediction_test_coreml.pkl'
tinyimagenet_resnet_prediction_train_coreml = 'tinyimagenet_resnet_prediction_train_coreml.pkl'

