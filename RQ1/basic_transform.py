import cv2
import random
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


def augmentation_width_shift(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(width_shift_range=0.2)
    it = datagen.flow(samples, batch_size=1)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_height_shift(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(height_shift_range=0.2)
    it = datagen.flow(samples, batch_size=1)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_horizontal_flip(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(horizontal_flip=True)
    it = datagen.flow(samples, batch_size=1)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_vertical_flip(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(vertical_flip=True)
    it = datagen.flow(samples, batch_size=1)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_rotation(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(rotation_range=45)
    it = datagen.flow(samples, batch_size=1)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_brightness(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(brightness_range=[0.2, 3.0])
    it = datagen.flow(samples, batch_size=1)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_zoom(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
    it = datagen.flow(samples, batch_size=1)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_featurewise_std_normalization(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(featurewise_std_normalization=True)
    it = datagen.flow(samples, batch_size=1)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_zca_whitening(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(zca_whitening=True)
    it = datagen.flow(samples, batch_size=1)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_shear_range(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(shear_range=0.5)
    it = datagen.flow(samples, batch_size=1, seed=8)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_channel_shift_range(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(channel_shift_range=80)
    it = datagen.flow(samples, batch_size=1)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def noise_random(image, noise_num=500):
    img_noise = image
    rows, cols, chn = img_noise.shape
    for i in range(noise_num):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img_noise[x, y, :] = 255
    return img_noise


def noise_salt_pepper(image, prob=0.03):
    output = np.zeros(image.shape, np.uint8)
    noise_out = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
                noise_out[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
                noise_out[i][j] = 255
            else:
                output[i][j] = image[i][j]
                noise_out[i][j] = 100
    return output


def noise_gasuss(image, mean=0, var=0.001):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out


def contrast(img):
    img_ = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    return img_


def run(img_np, select_method):
    if select_method == 'augmentation_width_shift':
        method = augmentation_width_shift
    elif select_method == 'augmentation_height_shift':
        method = augmentation_height_shift
    elif select_method == 'augmentation_horizontal_flip':
        method = augmentation_horizontal_flip
    elif select_method == 'augmentation_vertical_flip':
        method = augmentation_vertical_flip
    elif select_method == 'augmentation_rotation':
        method = augmentation_rotation
    elif select_method == 'augmentation_brightness':
        method = augmentation_brightness
    elif select_method == 'augmentation_zoom':
        method = augmentation_zoom
    elif select_method == 'augmentation_featurewise_std_normalization':
        method = augmentation_featurewise_std_normalization
    elif select_method == 'augmentation_zca_whitening':
        method = augmentation_zca_whitening
    elif select_method == 'augmentation_shear_range':
        method = augmentation_shear_range
    elif select_method == 'augmentation_channel_shift_range':
        method = augmentation_channel_shift_range
    elif select_method == 'noise_random':
        method = noise_random
    elif select_method == 'noise_salt_pepper':
        method = noise_salt_pepper
    elif select_method == 'noise_gasuss':
        method = noise_gasuss
    elif select_method == 'contrast':
        method = contrast

    new_img_list = []
    for img in img_np:
        tmp_img = method(img)
        new_img_list.append(tmp_img)
    new_img_np = np.array(new_img_list)
    return new_img_np

