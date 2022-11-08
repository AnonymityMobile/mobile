num_classes = 10

path_pre_data_lenet1_tflite_train = [
    ['augmentation_brightness_train.pkl', 'prediction_augmentation_brightness_x_train_lenet1_tflite.pkl'],
    ['augmentation_channel_shift_range_train.pkl', 'prediction_augmentation_channel_shift_range_x_train_lenet1_tflite.pkl'],
    ['augmentation_featurewise_std_normalization_train.pkl', 'prediction_augmentation_featurewise_std_normalization_x_train_lenet1_tflite.pkl'],
    ['augmentation_height_shift_train.pkl', 'prediction_augmentation_height_shift_x_train_lenet1_tflite.pkl'],
    ['augmentation_horizontal_flip_train.pkl', 'prediction_augmentation_horizontal_flip_x_train_lenet1_tflite.pkl'],
    ['augmentation_rotation_train.pkl', 'prediction_augmentation_rotation_x_train_lenet1_tflite.pkl'],
    ['augmentation_shear_range_train.pkl', 'prediction_augmentation_shear_range_x_train_lenet1_tflite.pkl'],
    ['augmentation_vertical_flip_train.pkl', 'prediction_augmentation_vertical_flip_x_train_lenet1_tflite.pkl'],
    ['augmentation_width_shift_train.pkl', 'prediction_augmentation_width_shift_x_train_lenet1_tflite.pkl'],
    ['augmentation_zca_whitening_train.pkl', 'prediction_augmentation_zca_whitening_x_train_lenet1_tflite.pkl'],
    ['augmentation_zoom_train.pkl', 'prediction_augmentation_zoom_x_train_lenet1_tflite.pkl'],
    ['contrast_train.pkl', 'prediction_contrast_x_train_lenet1_tflite.pkl'],
    ['noise_gasuss_train.pkl', 'prediction_noise_gasuss_x_train_lenet1_tflite.pkl'],
    ['noise_random_train.pkl', 'prediction_noise_random_x_train_lenet1_tflite.pkl'],
    ['noise_salt_pepper_train.pkl', 'prediction_noise_salt_pepper_x_train_lenet1_tflite.pkl'],
    ['x_train.pkl', 'prediction_x_train_lenet1_tflite.pkl']
]


path_pre_data_lenet1_tflite_test = [
    ['augmentation_brightness_test.pkl', 'prediction_augmentation_brightness_x_test_lenet1_tflite.pkl'],
    ['augmentation_channel_shift_range_test.pkl', 'prediction_augmentation_channel_shift_range_x_test_lenet1_tflite.pkl'],
    ['augmentation_featurewise_std_normalization_test.pkl', 'prediction_augmentation_featurewise_std_normalization_x_test_lenet1_tflite.pkl'],
    ['augmentation_height_shift_test.pkl', 'prediction_augmentation_height_shift_x_test_lenet1_tflite.pkl'],
    ['augmentation_horizontal_flip_test.pkl', 'prediction_augmentation_horizontal_flip_x_test_lenet1_tflite.pkl'],
    ['augmentation_rotation_test.pkl', 'prediction_augmentation_rotation_x_test_lenet1_tflite.pkl'],
    ['augmentation_shear_range_test.pkl', 'prediction_augmentation_shear_range_x_test_lenet1_tflite.pkl'],
    ['augmentation_vertical_flip_test.pkl', 'prediction_augmentation_vertical_flip_x_test_lenet1_tflite.pkl'],
    ['augmentation_width_shift_test.pkl', 'prediction_augmentation_width_shift_x_test_lenet1_tflite.pkl'],
    ['augmentation_zca_whitening_test.pkl', 'prediction_augmentation_zca_whitening_x_test_lenet1_tflite.pkl'],
    ['augmentation_zoom_test.pkl', 'prediction_augmentation_zoom_x_test_lenet1_tflite.pkl'],
    ['contrast_test.pkl', 'prediction_contrast_x_test_lenet1_tflite.pkl'],
    ['noise_gasuss_test.pkl', 'prediction_noise_gasuss_x_test_lenet1_tflite.pkl'],
    ['noise_random_test.pkl', 'prediction_noise_random_x_test_lenet1_tflite.pkl'],
    ['noise_salt_pepper_test.pkl', 'prediction_noise_salt_pepper_x_test_lenet1_tflite.pkl'],
    ['x_test.pkl', 'prediction_x_test_lenet1_tflite.pkl']
]





path_lenet1_coreml = 'mnist_prediction_lenet1_coreml/'
path_pre_data_lenet1_coreml_test = [
    ['augmentation_brightness_test.pkl', path_lenet1_coreml + 'prediction_augmentation_brightness_test_lenet1_coreml.pkl'],
    ['augmentation_channel_shift_range_test.pkl', path_lenet1_coreml + 'prediction_augmentation_channel_shift_range_test_lenet1_coreml.pkl'],
    ['augmentation_featurewise_std_normalization_test.pkl', path_lenet1_coreml + 'prediction_augmentation_featurewise_std_normalization_test_lenet1_coreml.pkl'],
    ['augmentation_height_shift_test.pkl', path_lenet1_coreml + 'prediction_augmentation_height_shift_test_lenet1_coreml.pkl'],
    ['augmentation_horizontal_flip_test.pkl', path_lenet1_coreml + 'prediction_augmentation_horizontal_flip_test_lenet1_coreml.pkl'],
    ['augmentation_rotation_test.pkl', path_lenet1_coreml + 'prediction_augmentation_rotation_test_lenet1_coreml.pkl'],
    ['augmentation_shear_range_test.pkl', path_lenet1_coreml + 'prediction_augmentation_shear_range_test_lenet1_coreml.pkl'],
    ['augmentation_vertical_flip_test.pkl', path_lenet1_coreml + 'prediction_augmentation_vertical_flip_test_lenet1_coreml.pkl'],
    ['augmentation_width_shift_test.pkl', path_lenet1_coreml + 'prediction_augmentation_width_shift_test_lenet1_coreml.pkl'],
    ['augmentation_zca_whitening_test.pkl', path_lenet1_coreml + 'prediction_augmentation_zca_whitening_test_lenet1_coreml.pkl'],
    ['augmentation_zoom_test.pkl', path_lenet1_coreml + 'prediction_augmentation_zoom_test_lenet1_coreml.pkl'],
    ['contrast_test.pkl', path_lenet1_coreml + 'prediction_contrast_test_lenet1_coreml.pkl'],
    ['noise_gasuss_test.pkl', path_lenet1_coreml + 'prediction_noise_gasuss_test_lenet1_coreml.pkl'],
    ['noise_random_test.pkl', path_lenet1_coreml + 'prediction_noise_random_test_lenet1_coreml.pkl'],
    ['noise_salt_pepper_test.pkl', path_lenet1_coreml + 'prediction_noise_salt_pepper_test_lenet1_coreml.pkl'],
    ['x_test.pkl', path_lenet1_coreml + 'prediction_x_test_lenet1_coreml.pkl']
]



path_pre_data_lenet1_coreml_train = [
    ['augmentation_brightness_train.pkl', path_lenet1_coreml + 'prediction_augmentation_brightness_train_lenet1_coreml.pkl'],
    ['augmentation_channel_shift_range_train.pkl', path_lenet1_coreml + 'prediction_augmentation_channel_shift_range_train_lenet1_coreml.pkl'],
    ['augmentation_featurewise_std_normalization_train.pkl', path_lenet1_coreml + 'prediction_augmentation_featurewise_std_normalization_train_lenet1_coreml.pkl'],
    ['augmentation_height_shift_train.pkl', path_lenet1_coreml + 'prediction_augmentation_height_shift_train_lenet1_coreml.pkl'],
    ['augmentation_horizontal_flip_train.pkl', path_lenet1_coreml + 'prediction_augmentation_horizontal_flip_train_lenet1_coreml.pkl'],
    ['augmentation_rotation_train.pkl', path_lenet1_coreml + 'prediction_augmentation_rotation_train_lenet1_coreml.pkl'],
    ['augmentation_shear_range_train.pkl', path_lenet1_coreml + 'prediction_augmentation_shear_range_train_lenet1_coreml.pkl'],
    ['augmentation_vertical_flip_train.pkl', path_lenet1_coreml + 'prediction_augmentation_vertical_flip_train_lenet1_coreml.pkl'],
    ['augmentation_width_shift_train.pkl', path_lenet1_coreml + 'prediction_augmentation_width_shift_train_lenet1_coreml.pkl'],
    ['augmentation_zca_whitening_train.pkl', path_lenet1_coreml + 'prediction_augmentation_zca_whitening_train_lenet1_coreml.pkl'],
    ['augmentation_zoom_train.pkl', path_lenet1_coreml + 'prediction_augmentation_zoom_train_lenet1_coreml.pkl'],
    ['contrast_train.pkl', path_lenet1_coreml + 'prediction_contrast_train_lenet1_coreml.pkl'],
    ['noise_gasuss_train.pkl', path_lenet1_coreml + 'prediction_noise_gasuss_train_lenet1_coreml.pkl'],
    ['noise_random_train.pkl', path_lenet1_coreml + 'prediction_noise_random_train_lenet1_coreml.pkl'],
    ['noise_salt_pepper_train.pkl', path_lenet1_coreml + 'prediction_noise_salt_pepper_train_lenet1_coreml.pkl'],
    ['x_train.pkl', path_lenet1_coreml + 'prediction_x_train_lenet1_coreml.pkl']
]



path_pre_data_lenet5_tflite_train = [
    ['augmentation_brightness_train.pkl', 'prediction_augmentation_brightness_train_lenet5_tflite.pkl'],
    ['augmentation_channel_shift_range_train.pkl', 'prediction_augmentation_channel_shift_range_train_lenet5_tflite.pkl'],
    ['augmentation_featurewise_std_normalization_train.pkl', 'prediction_augmentation_featurewise_std_normalization_train_lenet5_tflite.pkl'],
    ['augmentation_height_shift_train.pkl', 'prediction_augmentation_height_shift_train_lenet5_tflite.pkl'],
    ['augmentation_horizontal_flip_train.pkl', 'prediction_augmentation_horizontal_flip_train_lenet5_tflite.pkl'],
    ['augmentation_rotation_train.pkl', 'prediction_augmentation_rotation_train_lenet5_tflite.pkl'],
    ['augmentation_shear_range_train.pkl', 'prediction_augmentation_shear_range_train_lenet5_tflite.pkl'],
    ['augmentation_vertical_flip_train.pkl', 'prediction_augmentation_vertical_flip_train_lenet5_tflite.pkl'],
    ['augmentation_width_shift_train.pkl', 'prediction_augmentation_width_shift_train_lenet5_tflite.pkl'],
    ['augmentation_zca_whitening_train.pkl', 'prediction_augmentation_zca_whitening_train_lenet5_tflite.pkl'],
    ['augmentation_zoom_train.pkl', 'prediction_augmentation_zoom_train_lenet5_tflite.pkl'],
    ['contrast_train.pkl', 'prediction_contrast_train_lenet5_tflite.pkl'],
    ['noise_gasuss_train.pkl', 'prediction_noise_gasuss_train_lenet5_tflite.pkl'],
    ['noise_random_train.pkl', 'prediction_noise_random_train_lenet5_tflite.pkl'],
    ['noise_salt_pepper_train.pkl', 'prediction_noise_salt_pepper_train_lenet5_tflite.pkl'],
    ['x_train.pkl', 'prediction_x_train_lenet5_tflite.pkl']
]


path_pre_data_lenet5_tflite_test = [
    ['augmentation_brightness_test.pkl', 'prediction_augmentation_brightness_test_lenet5_tflite.pkl'],
    ['augmentation_channel_shift_range_test.pkl', 'prediction_augmentation_channel_shift_range_test_lenet5_tflite.pkl'],
    ['augmentation_featurewise_std_normalization_test.pkl', 'prediction_augmentation_featurewise_std_normalization_test_lenet5_tflite.pkl'],
    ['augmentation_height_shift_test.pkl', 'prediction_augmentation_height_shift_test_lenet5_tflite.pkl'],
    ['augmentation_horizontal_flip_test.pkl', 'prediction_augmentation_horizontal_flip_test_lenet5_tflite.pkl'],
    ['augmentation_rotation_test.pkl', 'prediction_augmentation_rotation_test_lenet5_tflite.pkl'],
    ['augmentation_shear_range_test.pkl', 'prediction_augmentation_shear_range_test_lenet5_tflite.pkl'],
    ['augmentation_vertical_flip_test.pkl', 'prediction_augmentation_vertical_flip_test_lenet5_tflite.pkl'],
    ['augmentation_width_shift_test.pkl', 'prediction_augmentation_width_shift_test_lenet5_tflite.pkl'],
    ['augmentation_zca_whitening_test.pkl', 'prediction_augmentation_zca_whitening_test_lenet5_tflite.pkl'],
    ['augmentation_zoom_test.pkl', 'prediction_augmentation_zoom_test_lenet5_tflite.pkl'],
    ['contrast_test.pkl', 'prediction_contrast_test_lenet5_tflite.pkl'],
    ['noise_gasuss_test.pkl', 'prediction_noise_gasuss_test_lenet5_tflite.pkl'],
    ['noise_random_test.pkl', 'prediction_noise_random_test_lenet5_tflite.pkl'],
    ['noise_salt_pepper_test.pkl', 'prediction_noise_salt_pepper_test_lenet5_tflite.pkl'],
    ['x_test.pkl', 'prediction_x_test_lenet5_tflite.pkl']
]




path_lenet5_coreml = 'mnist_prediction_lenet5_coreml/'
path_pre_data_lenet5_coreml_test = [
    ['augmentation_brightness_test.pkl', path_lenet5_coreml + 'prediction_augmentation_brightness_test_lenet5_coreml.pkl'],
    ['augmentation_channel_shift_range_test.pkl', path_lenet5_coreml + 'prediction_augmentation_channel_shift_range_test_lenet5_coreml.pkl'],
    ['augmentation_featurewise_std_normalization_test.pkl', path_lenet5_coreml + 'prediction_augmentation_featurewise_std_normalization_test_lenet5_coreml.pkl'],
    ['augmentation_height_shift_test.pkl', path_lenet5_coreml + 'prediction_augmentation_height_shift_test_lenet5_coreml.pkl'],
    ['augmentation_horizontal_flip_test.pkl', path_lenet5_coreml + 'prediction_augmentation_horizontal_flip_test_lenet5_coreml.pkl'],
    ['augmentation_rotation_test.pkl', path_lenet5_coreml + 'prediction_augmentation_rotation_test_lenet5_coreml.pkl'],
    ['augmentation_shear_range_test.pkl', path_lenet5_coreml + 'prediction_augmentation_shear_range_test_lenet5_coreml.pkl'],
    ['augmentation_vertical_flip_test.pkl', path_lenet5_coreml + 'prediction_augmentation_vertical_flip_test_lenet5_coreml.pkl'],
    ['augmentation_width_shift_test.pkl', path_lenet5_coreml + 'prediction_augmentation_width_shift_test_lenet5_coreml.pkl'],
    ['augmentation_zca_whitening_test.pkl', path_lenet5_coreml + 'prediction_augmentation_zca_whitening_test_lenet5_coreml.pkl'],
    ['augmentation_zoom_test.pkl', path_lenet5_coreml + 'prediction_augmentation_zoom_test_lenet5_coreml.pkl'],
    ['contrast_test.pkl', path_lenet5_coreml + 'prediction_contrast_test_lenet5_coreml.pkl'],
    ['noise_gasuss_test.pkl', path_lenet5_coreml + 'prediction_noise_gasuss_test_lenet5_coreml.pkl'],
    ['noise_random_test.pkl', path_lenet5_coreml + 'prediction_noise_random_test_lenet5_coreml.pkl'],
    ['noise_salt_pepper_test.pkl', path_lenet5_coreml + 'prediction_noise_salt_pepper_test_lenet5_coreml.pkl'],
    ['x_test.pkl', path_lenet5_coreml + 'prediction_x_test_lenet5_coreml.pkl']
]



path_pre_data_lenet5_coreml_train = [
    ['augmentation_brightness_train.pkl', path_lenet5_coreml + 'prediction_augmentation_brightness_train_lenet5_coreml.pkl'],
    ['augmentation_channel_shift_range_train.pkl', path_lenet5_coreml + 'prediction_augmentation_channel_shift_range_train_lenet5_coreml.pkl'],
    ['augmentation_featurewise_std_normalization_train.pkl', path_lenet5_coreml + 'prediction_augmentation_featurewise_std_normalization_train_lenet5_coreml.pkl'],
    ['augmentation_height_shift_train.pkl', path_lenet5_coreml + 'prediction_augmentation_height_shift_train_lenet5_coreml.pkl'],
    ['augmentation_horizontal_flip_train.pkl', path_lenet5_coreml + 'prediction_augmentation_horizontal_flip_train_lenet5_coreml.pkl'],
    ['augmentation_rotation_train.pkl', path_lenet5_coreml + 'prediction_augmentation_rotation_train_lenet5_coreml.pkl'],
    ['augmentation_shear_range_train.pkl', path_lenet5_coreml + 'prediction_augmentation_shear_range_train_lenet5_coreml.pkl'],
    ['augmentation_vertical_flip_train.pkl', path_lenet5_coreml + 'prediction_augmentation_vertical_flip_train_lenet5_coreml.pkl'],
    ['augmentation_width_shift_train.pkl', path_lenet5_coreml + 'prediction_augmentation_width_shift_train_lenet5_coreml.pkl'],
    ['augmentation_zca_whitening_train.pkl', path_lenet5_coreml + 'prediction_augmentation_zca_whitening_train_lenet5_coreml.pkl'],
    ['augmentation_zoom_train.pkl', path_lenet5_coreml + 'prediction_augmentation_zoom_train_lenet5_coreml.pkl'],
    ['contrast_train.pkl', path_lenet5_coreml + 'prediction_contrast_train_lenet5_coreml.pkl'],
    ['noise_gasuss_train.pkl', path_lenet5_coreml + 'prediction_noise_gasuss_train_lenet5_coreml.pkl'],
    ['noise_random_train.pkl', path_lenet5_coreml + 'prediction_noise_random_train_lenet5_coreml.pkl'],
    ['noise_salt_pepper_train.pkl', path_lenet5_coreml + 'prediction_noise_salt_pepper_train_lenet5_coreml.pkl'],
    ['x_train.pkl', path_lenet5_coreml + 'prediction_x_train_lenet5_coreml.pkl']
]