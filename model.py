from __future__ import absolute_import, division, print_function, unicode_literals

import math
import os
import random
from glob import glob

import elasticdeform
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tifffile
from keras.src.callbacks import Callback, CSVLogger, ModelCheckpoint
from keras.src.layers import Conv3D
from keras.src.optimizers import Adam, SGD, RMSprop
from skimage import transform
from tensorflow.python.keras.utils.data_utils import Sequence

print("TensorFlow version: {}".format(tf.__version__))

from keras import backend as K, Model

from keras.src.layers import BatchNormalization
from keras.src.layers import ReLU
from keras.src.layers import MaxPooling3D
from keras.src.layers import Conv3DTranspose
from keras.src.layers import Input
from keras.src.layers import Concatenate

Notebook_version = '2.2.1'
Network = 'U-Net (3D)'

# Create a variable to get and store relative base path
base_path = os.getcwd()

# Define MultiPageTiffGenerator class
class MultiPageTiffGenerator(Sequence):

    def __init__(self,
                 source_path,
                 target_path,
                 batch_size=1,
                 shape=(128, 128, 32, 1),
                 augment=False,
                 augmentations=[],
                 deform_augment=False,
                 deform_augmentation_params=(5, 3, 4),
                 val_split=0.2,
                 is_val=False,
                 random_crop=True,
                 downscale=1,
                 binary_target=False):

        # If directory with various multi-page tiffiles is provided read as list
        if os.path.isfile(source_path):
            self.dir_flag = False
            self.source = tifffile.imread(source_path)
            if binary_target:
                self.target = tifffile.imread(target_path).astype(bool)
            else:
                self.target = tifffile.imread(target_path)

        elif os.path.isdir(source_path):
            self.dir_flag = True
            self.source_dir_list = glob(os.path.join(source_path, '*'))
            self.target_dir_list = glob(os.path.join(target_path, '*'))

            self.source_dir_list.sort()
            self.target_dir_list.sort()

        self.shape = shape
        self.batch_size = batch_size
        self.augment = augment
        self.val_split = val_split
        self.is_val = is_val
        self.random_crop = random_crop
        self.downscale = downscale
        self.binary_target = binary_target
        self.deform_augment = deform_augment
        self.on_epoch_end()

        if self.augment:
            # pass list of augmentation functions 
            self.seq = iaa.Sequential(augmentations, random_order=True)  # apply augmenters in random order
        if self.deform_augment:
            self.deform_sigma, self.deform_points, self.deform_order = deform_augmentation_params

    def __len__(self):
        # If various multi-page tiff files provided sum all images within each
        if self.augment:
            augment_factor = 4
        else:
            augment_factor = 1

        if self.dir_flag:
            num_of_imgs = 0
            for tiff_path in self.source_dir_list:
                num_of_imgs += tifffile.imread(tiff_path).shape[0]
            xy_shape = tifffile.imread(self.source_dir_list[0]).shape[1:]

            if self.is_val:
                if self.random_crop:
                    crop_volume = self.shape[0] * self.shape[1] * self.shape[2]
                    volume = xy_shape[0] * xy_shape[1] * self.val_split * num_of_imgs
                    return math.floor(augment_factor * volume / (crop_volume * self.batch_size * self.downscale))
                else:
                    return math.floor(self.val_split * num_of_imgs / self.batch_size)
            else:
                if self.random_crop:
                    crop_volume = self.shape[0] * self.shape[1] * self.shape[2]
                    volume = xy_shape[0] * xy_shape[1] * (1 - self.val_split) * num_of_imgs
                    return math.floor(augment_factor * volume / (crop_volume * self.batch_size * self.downscale))

                else:
                    return math.floor(augment_factor * (1 - self.val_split) * num_of_imgs / self.batch_size)
        else:
            if self.is_val:
                if self.random_crop:
                    crop_volume = self.shape[0] * self.shape[1] * self.shape[2]
                    volume = self.source.shape[0] * self.source.shape[1] * self.val_split * self.source.shape[2]
                    return math.floor(augment_factor * volume / (crop_volume * self.batch_size * self.downscale))
                else:
                    return math.floor((self.val_split * self.source.shape[0] / self.batch_size))
            else:
                if self.random_crop:
                    crop_volume = self.shape[0] * self.shape[1] * self.shape[2]
                    volume = self.source.shape[0] * self.source.shape[1] * (1 - self.val_split) * self.source.shape[2]
                    return math.floor(augment_factor * volume / (crop_volume * self.batch_size * self.downscale))
                else:
                    return math.floor(augment_factor * (1 - self.val_split) * self.source.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        source_batch = np.empty((self.batch_size,
                                 self.shape[0],
                                 self.shape[1],
                                 self.shape[2],
                                 self.shape[3]))
        target_batch = np.empty((self.batch_size,
                                 self.shape[0],
                                 self.shape[1],
                                 self.shape[2],
                                 self.shape[3]))

        for batch in range(self.batch_size):
            # Modulo operator ensures IndexError is avoided
            stack_start = self.batch_list[(idx + batch * self.shape[2]) % len(self.batch_list)]

            if self.dir_flag:
                self.source = tifffile.imread(self.source_dir_list[stack_start[0]])
                if self.binary_target:
                    self.target = tifffile.imread(self.target_dir_list[stack_start[0]]).astype(bool)
                else:
                    self.target = tifffile.imread(self.target_dir_list[stack_start[0]])

            src_list = []
            tgt_list = []
            for i in range(stack_start[1], stack_start[1] + self.shape[2]):
                src = self.source[i]
                src = transform.downscale_local_mean(src, (self.downscale, self.downscale))
                if not self.random_crop:
                    src = transform.resize(src, (self.shape[0], self.shape[1]), mode='constant', preserve_range=True)
                src = self._min_max_scaling(src)
                src_list.append(src)

                tgt = self.target[i]
                tgt = transform.downscale_local_mean(tgt, (self.downscale, self.downscale))
                if not self.random_crop:
                    tgt = transform.resize(tgt, (self.shape[0], self.shape[1]), mode='constant', preserve_range=True)
                if not self.binary_target:
                    tgt = self._min_max_scaling(tgt)
                tgt_list.append(tgt)

            if self.random_crop:
                if src.shape[0] == self.shape[0]:
                    x_rand = 0
                if src.shape[1] == self.shape[1]:
                    y_rand = 0
                if src.shape[0] > self.shape[0]:
                    x_rand = np.random.randint(src.shape[0] - self.shape[0])
                if src.shape[1] > self.shape[1]:
                    y_rand = np.random.randint(src.shape[1] - self.shape[1])
                if src.shape[0] < self.shape[0] or src.shape[1] < self.shape[1]:
                    raise ValueError('Patch shape larger than (downscaled) source shape')

            for i in range(self.shape[2]):
                if self.random_crop:
                    src = src_list[i]
                    tgt = tgt_list[i]
                    src_crop = src[x_rand:self.shape[0] + x_rand, y_rand:self.shape[1] + y_rand]
                    tgt_crop = tgt[x_rand:self.shape[0] + x_rand, y_rand:self.shape[1] + y_rand]
                else:
                    src_crop = src_list[i]
                    tgt_crop = tgt_list[i]

                source_batch[batch, :, :, i, 0] = src_crop
                target_batch[batch, :, :, i, 0] = tgt_crop

        if self.augment:
            # On-the-fly data augmentation
            source_batch, target_batch = self.augment_volume(source_batch, target_batch)

            # Data augmentation by reversing stack
            if np.random.random() > 0.5:
                source_batch, target_batch = source_batch[::-1], target_batch[::-1]

            # Data augmentation by elastic deformation
            if np.random.random() > 0.5 and self.deform_augment:
                source_batch, target_batch = self.deform_volume(source_batch, target_batch)

            if not self.binary_target:
                target_batch = self._min_max_scaling(target_batch)

            return self._min_max_scaling(source_batch), target_batch

        else:
            return source_batch, target_batch

    def on_epoch_end(self):
        # Validation split performed here
        self.batch_list = []
        # Create batch_list of all combinations of tifffile and stack position
        if self.dir_flag:
            for i in range(len(self.source_dir_list)):
                num_of_pages = tifffile.imread(self.source_dir_list[i]).shape[0]
                if self.is_val:
                    start_page = num_of_pages - math.floor(self.val_split * num_of_pages)
                    for j in range(start_page, num_of_pages - self.shape[2]):
                        self.batch_list.append([i, j])
                else:
                    last_page = math.floor((1 - self.val_split) * num_of_pages)
                    for j in range(last_page - self.shape[2]):
                        self.batch_list.append([i, j])
        else:
            num_of_pages = self.source.shape[0]
            if self.is_val:
                start_page = num_of_pages - math.floor(self.val_split * num_of_pages)
                for j in range(start_page, num_of_pages - self.shape[2]):
                    self.batch_list.append([0, j])

            else:
                last_page = math.floor((1 - self.val_split) * num_of_pages)
                for j in range(last_page - self.shape[2]):
                    self.batch_list.append([0, j])

        if self.is_val and (len(self.batch_list) <= 0):
            raise ValueError('validation_split too small! Increase val_split or decrease z-depth')
        random.shuffle(self.batch_list)

    def _min_max_scaling(self, data):
        n = data - np.min(data)
        d = np.max(data) - np.min(data)

        return n / d

    def class_weights(self):
        ones = 0
        pixels = 0

        if self.dir_flag:
            for i in range(len(self.target_dir_list)):
                tgt = tifffile.imread(self.target_dir_list[i]).astype(bool)
                ones += np.sum(tgt)
                pixels += tgt.shape[0] * tgt.shape[1] * tgt.shape[2]
        else:
            ones = np.sum(self.target)
            pixels = self.target.shape[0] * self.target.shape[1] * self.target.shape[2]
        p_ones = ones / pixels
        p_zeros = 1 - p_ones

        # Return swapped probability to increase weight of unlikely class
        return p_ones, p_zeros

    def deform_volume(self, src_vol, tgt_vol):
        [src_dfrm, tgt_dfrm] = elasticdeform.deform_random_grid([src_vol, tgt_vol],
                                                                axis=(1, 2, 3),
                                                                sigma=self.deform_sigma,
                                                                points=self.deform_points,
                                                                order=self.deform_order)
        if self.binary_target:
            tgt_dfrm = tgt_dfrm > 0.1

        return self._min_max_scaling(src_dfrm), tgt_dfrm

    def augment_volume(self, src_vol, tgt_vol):

        src_vol_aug = np.empty(src_vol.shape)
        tgt_vol_aug = np.empty(tgt_vol.shape)

        for i in range(src_vol.shape[3]):
            src_aug_z, tgt_aug_z = self.seq(images=src_vol[:, :, :, i, 0].astype('float16'),
                                            segmentation_maps=np.expand_dims(tgt_vol[:, :, :, i, 0].astype(bool),
                                                                             axis=-1))
            src_vol_aug[:, :, :, i, 0] = src_aug_z
            tgt_vol_aug[:, :, :, i, 0] = np.squeeze(tgt_aug_z)
        return self._min_max_scaling(src_vol_aug), tgt_vol_aug

    def sample_augmentation(self, idx):
        src, tgt = self.__getitem__(idx)

        src_aug, tgt_aug = self.augment_volume(src, tgt)

        if self.deform_augment:
            src_aug, tgt_aug = self.deform_volume(src_aug, tgt_aug)

        return src_aug, tgt_aug

    # Define custom loss and dice coefficient


def dice_coefficient(y_true, y_pred):
    eps = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps)


def weighted_binary_crossentropy(zero_weight, one_weight):
    def _weighted_binary_crossentropy(y_true, y_pred):
        binary_crossentropy = K.binary_crossentropy(y_true, y_pred)

        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_binary_crossentropy = weight_vector * binary_crossentropy

        return K.mean(weighted_binary_crossentropy)

    return _weighted_binary_crossentropy


# Custom callback showing sample prediction
class SampleImageCallback(Callback):

    def __init__(self, model, sample_data, model_path, save=False):
        super().__init__()
        self.model = model
        self.sample_data = sample_data
        self.model_path = model_path
        self.save = save

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        sample_predict = self.model.predict_on_batch(self.sample_data)

        f = plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(self.sample_data[0, :, :, 0, 0], interpolation='nearest', cmap='gray')
        plt.title('Sample source')
        plt.axis('off');

        plt.subplot(1, 2, 2)
        plt.imshow(sample_predict[0, :, :, 0, 0], interpolation='nearest', cmap='magma')
        plt.title('Predicted target')
        plt.axis('off');

        plt.show()

        if self.save:
            plt.savefig(self.model_path + '/epoch_' + str(epoch + 1) + '.png')


# Define Unet3D class
class Unet3D:

    def __init__(self,
                 shape=(256, 256, 16, 1)):
        if isinstance(shape, str):
            shape = eval(shape)

        self.shape = shape

        input_tensor = Input(self.shape, name='input')

        self.model = self.unet_3D(input_tensor)

    def down_block_3D(self, input_tensor, filters):
        x = Conv3D(filters=filters, kernel_size=(3, 3, 3), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv3D(filters=filters * 2, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    def up_block_3D(self, input_tensor, concat_layer, filters):
        x = Conv3DTranspose(filters, kernel_size=(2, 2, 2), strides=(2, 2, 2))(input_tensor)

        x = Concatenate()([x, concat_layer])

        x = Conv3D(filters=filters, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv3D(filters=filters * 2, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    def unet_3D(self, input_tensor, filters=32):
        d1 = self.down_block_3D(input_tensor, filters=filters)
        p1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_last')(d1)
        d2 = self.down_block_3D(p1, filters=filters * 2)
        p2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_last')(d2)
        d3 = self.down_block_3D(p2, filters=filters * 4)
        p3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_last')(d3)

        d4 = self.down_block_3D(p3, filters=filters * 8)

        u1 = self.up_block_3D(d4, d3, filters=filters * 4)
        u2 = self.up_block_3D(u1, d2, filters=filters * 2)
        u3 = self.up_block_3D(u2, d1, filters=filters)

        output_tensor = Conv3D(filters=1, kernel_size=(1, 1, 1), activation='sigmoid')(u3)

        return Model(inputs=[input_tensor], outputs=[output_tensor])

    def summary(self):
        return self.model.summary()

    # Pass generators instead
    def train(self,
              epochs,
              batch_size,
              train_generator,
              val_generator,
              model_path,
              model_name,
              optimizer='adam',
              learning_rate=0.001,
              loss='weighted_binary_crossentropy',
              metrics='dice',
              ckpt_period=1,
              save_best_ckpt_only=False,
              ckpt_path=None):

        class_weight_zero, class_weight_one = train_generator.class_weights()

        if loss == 'weighted_binary_crossentropy':
            loss = weighted_binary_crossentropy(class_weight_zero, class_weight_one)

        if metrics == 'dice':
            metrics = dice_coefficient

        if optimizer == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optimizer = SGD(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=[metrics])

        if ckpt_path is not None:
            self.model.load_weights(ckpt_path)

        full_model_path = os.path.join(model_path, model_name)

        if not os.path.exists(full_model_path):
            os.makedirs(full_model_path)

        log_dir = full_model_path + '/Quality Control'

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        ckpt_dir = full_model_path + '/ckpt'

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        csv_out_name = log_dir + '/training_evaluation.csv'
        if ckpt_path is None:
            csv_logger = CSVLogger(csv_out_name)
        else:
            csv_logger = CSVLogger(csv_out_name, append=True)

        if save_best_ckpt_only:
            ckpt_name = ckpt_dir + '/' + model_name + '.hdf5'
        else:
            ckpt_name = ckpt_dir + '/' + model_name + '_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.hdf5'

        model_ckpt = ModelCheckpoint(ckpt_name,
                                     verbose=1,
                                     save_freq=ckpt_period,
                                     save_best_only=save_best_ckpt_only,
                                     save_weights_only=True)

        sample_batch, __ = val_generator.__getitem__(random.randint(0, len(val_generator)))
        sample_img = SampleImageCallback(self.model,
                                         sample_batch,
                                         model_path)

        self.model.fit(train_generator,
                       validation_data=val_generator,
                       validation_steps=math.floor(len(val_generator) / batch_size),
                       epochs=epochs,
                       callbacks=[csv_logger,
                                  model_ckpt,
                                  sample_img])

        last_ckpt_name = ckpt_dir + '/' + model_name + '_last.hdf5'
        self.model.save_weights(last_ckpt_name)

    def _min_max_scaling(self, data):
        n = data - np.min(data)
        d = np.max(data) - np.min(data)

        return n / d

    def predict(self,
                input,
                ckpt_path,
                z_range=None,
                downscaling=None,
                true_patch_size=None):

        self.model.load_weights(ckpt_path)

        if isinstance(downscaling, str):
            downscaling = eval(downscaling)

        if math.isnan(downscaling):
            downscaling = None

        if isinstance(true_patch_size, str):
            true_patch_size = eval(true_patch_size)

        if not isinstance(true_patch_size, tuple):
            if math.isnan(true_patch_size):
                true_patch_size = None

        if isinstance(input, str):
            src_volume = tifffile.imread(input)
        elif isinstance(input, np.ndarray):
            src_volume = input
        else:
            raise TypeError('Input is not path or numpy array!')

        in_size = src_volume.shape

        if downscaling or true_patch_size is not None:
            x_scaling = 0
            y_scaling = 0

            if true_patch_size is not None:
                x_scaling += true_patch_size[0] / self.shape[0]
                y_scaling += true_patch_size[1] / self.shape[1]
            if downscaling is not None:
                x_scaling += downscaling
                y_scaling += downscaling

            src_list = []
            for i in range(src_volume.shape[0]):
                src_list.append(transform.downscale_local_mean(src_volume[i], (int(x_scaling), int(y_scaling))))
            src_volume = np.array(src_list)

        if z_range is not None:
            src_volume = src_volume[z_range[0]:z_range[1]]

        src_volume = self._min_max_scaling(src_volume)

        src_array = np.zeros((1,
                              math.ceil(src_volume.shape[1] / self.shape[0]) * self.shape[0],
                              math.ceil(src_volume.shape[2] / self.shape[1]) * self.shape[1],
                              math.ceil(src_volume.shape[0] / self.shape[2]) * self.shape[2],
                              self.shape[3]))

        for i in range(src_volume.shape[0]):
            src_array[0, :src_volume.shape[1], :src_volume.shape[2], i, 0] = src_volume[i]

        pred_array = np.empty(src_array.shape)
        print(src_volume.dtype)
        for i in range(math.ceil(src_volume.shape[1] / self.shape[0])):
            for j in range(math.ceil(src_volume.shape[2] / self.shape[1])):
                for k in range(math.ceil(src_volume.shape[0] / self.shape[2])):
                    pred_temp = self.model.predict(src_array[:,
                                                   i * self.shape[0]:i * self.shape[0] + self.shape[0],
                                                   j * self.shape[1]:j * self.shape[1] + self.shape[1],
                                                   k * self.shape[2]:k * self.shape[2] + self.shape[2]])
                    pred_array[:,
                    i * self.shape[0]:i * self.shape[0] + self.shape[0],
                    j * self.shape[1]:j * self.shape[1] + self.shape[1],
                    k * self.shape[2]:k * self.shape[2] + self.shape[2]] = pred_temp

        pred_volume = np.rollaxis(np.squeeze(pred_array), -1)[:src_volume.shape[0], :src_volume.shape[1],
                      :src_volume.shape[2]]

        if downscaling is not None:
            pred_list = []
            for i in range(pred_volume.shape[0]):
                pred_list.append(transform.resize(pred_volume[i], (in_size[1], in_size[2]), preserve_range=True))
            pred_volume = np.array(pred_list)

        return pred_volume
