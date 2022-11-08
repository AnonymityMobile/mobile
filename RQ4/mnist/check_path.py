import os
check_list = os.listdir('/mnt/irisgpfs/users/yili/pycharm/data/mnist_pkl')

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

for i in path_pre_data_lenet1_tflite_test:
    if i[0] and i[1] in check_list:
        print('yes')
    else:
        print('==============')


print(check_list)