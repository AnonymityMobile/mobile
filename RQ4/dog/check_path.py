import os
check_list = os.listdir('/mnt/irisgpfs/users/pycharm/data/dog_image_pkl')

path_pre_data_dogscanner = [
    ['augmentation_brightness_dog_image_data.pkl', 'prediction_dogscanner_augmentation_brightness_dog_image_data.pkl'],
    ['augmentation_channel_shift_range_dog_image_data.pkl', 'prediction_dogscanner_augmentation_channel_shift_range_dog_image_data.pkl'],
    ['augmentation_featurewise_std_normalization_dog_image_data.pkl', 'prediction_dogscanner_augmentation_featurewise_std_normalization_dog_image_data.pkl'],
    ['augmentation_height_shift_dog_image_data.pkl', 'prediction_dogscanner_augmentation_height_shift_dog_image_data.pkl'],
    ['augmentation_horizontal_flip_dog_image_data.pkl', 'prediction_dogscanner_augmentation_horizontal_flip_dog_image_data.pkl'],
    ['augmentation_rotation_dog_image_data.pkl', 'prediction_dogscanner_augmentation_rotation_dog_image_data.pkl'],
    ['augmentation_shear_range_dog_image_data.pkl', 'prediction_dogscanner_augmentation_shear_range_dog_image_data.pkl'],
    ['augmentation_vertical_flip_dog_image_data.pkl', 'prediction_dogscanner_augmentation_vertical_flip_dog_image_data.pkl'],
    ['augmentation_width_shift_dog_image_data.pkl', 'prediction_dogscanner_augmentation_width_shift_dog_image_data.pkl'],
    ['augmentation_zca_whitening_dog_image_data.pkl', 'prediction_dogscanner_augmentation_zca_whitening_dog_image_data.pkl'],
    ['augmentation_zoom_dog_image_data.pkl', 'prediction_dogscanner_augmentation_zoom_dog_image_data.pkl'],
    ['contrast_dog_image_data.pkl', 'prediction_dogscanner_contrast_dog_image_data.pkl'],
    ['noise_gasuss_dog_image_data.pkl', 'prediction_dogscanner_noise_gasuss_dog_image_data.pkl'],
    ['noise_random_dog_image_data.pkl', 'prediction_dogscanner_noise_random_dog_image_data.pkl'],
    ['noise_salt_pepper_dog_image_data.pkl', 'prediction_dogscanner_noise_salt_pepper_dog_image_data.pkl'],
    ['dog_image_data.pkl', 'prediction_dogscanner_dog_image_data.pkl']
]

for i in path_pre_data_dogscanner:
    if i[0] and i[1] in check_list:
        print('yes')
    else:
        print('==============')
