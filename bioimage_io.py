import os

import pandas as pd
from bioimageio.core import load_raw_resource_description
from zipfile import ZipFile
import requests
from matplotlib import pyplot as plt

from model import Unet3D
from skimage import transform
from skimage import exposure
from skimage import color
from skimage import io



model_name = ""
model_path = "zero/notebook_stardist_3d_zerocostdl4mic"

full_model_path = os.path.join(model_path, model_name)

# @markdown ---

# @markdown ###Training parameters
number_of_epochs = 2  # @param {type:"number"}

# @markdown ###Default advanced parameters
use_default_advanced_parameters = False  # @param {type:"boolean"}

# @markdown <font size = 3>If not, please change:

batch_size = 1  # @param {type:"number"}
patch_size = (512, 512, 8)  # @param {type:"number"} # in pixels
training_shape = patch_size + (1,)
image_pre_processing = 'randomly crop to patch_size'  # @param ["randomly crop to patch_size", "resize to patch_size"]

validation_split_in_percent = 20  # @param{type:"number"}
downscaling_in_xy = 1  # @param {type:"number"} # in pixels

binary_target = True  # @param {type:"boolean"}

loss_function = 'weighted_binary_crossentropy'  # @param ["weighted_binary_crossentropy", "binary_crossentropy", "categorical_crossentropy", "sparse_categorical_crossentropy", "mean_squared_error", "mean_absolute_error"]

metrics = 'dice'  # @param ["dice", "accuracy"]

optimizer = 'adam'  # @param ["adam", "sgd", "rmsprop"]

learning_rate = 0.001  # @param{type:"number"}

if image_pre_processing == "randomly crop to patch_size":
    random_crop = True
else:
    random_crop = False

if use_default_advanced_parameters:
    print("Default advanced parameters enabled")
    batch_size = 3
    training_shape = (256, 256, 8, 1)
    validation_split_in_percent = 20
    downscaling_in_xy = 1
    random_crop = True
    binary_target = True
    loss_function = 'weighted_binary_crossentropy'
    metrics = 'dice'
    optimizer = 'adam'
    learning_rate = 0.001
# @markdown ###Checkpointing parameters
# checkpointing_period = 1 #@param {type:"number"}
checkpointing_period = "epoch"
# @markdown  <font size = 3>If chosen, only the best checkpoint is saved. Otherwise a checkpoint is saved every epoch:
save_best_only = False  # @param {type:"boolean"}

# @markdown ###Resume training
# @markdown <font size = 3>Choose if training was interrupted:
resume_training = False  # @param {type:"boolean"}

# @markdown ###Transfer learning
# @markdown <font size = 3>For transfer learning, do not select resume_training and specify a checkpoint_path below.

# @markdown <font size = 3> - If the model is already downloaded or is locally available, please specify the path to the .h5 file.

# @markdown <font size = 3> - To use a model from the BioImage Model Zoo, write the model ID. For example: 10.5281/zenodo.5749843

pretrained_model_choice = "bioimageio_model"  # @param ["Model_from_file", "bioimageio_model"]
checkpoint_path = ""  # @param {type:"string"}
model_id = ""  # @param {type:"string"}
# --------------------- Load the model from a bioimageio model (can be path on drive or url / doi) ---
if pretrained_model_choice == "bioimageio_model":
    from bioimageio.core import load_raw_resource_description
    from zipfile import ZipFile
    import requests

    model_spec = load_raw_resource_description(model_id)
    if "keras_hdf5" not in model_spec.weights:
        print("Invalid bioimageio model")
    else:
        url = model_spec.weights["keras_hdf5"].source
        r = requests.get(url, allow_redirects=True)
        open("keras_model.h5", 'wb').write(r.content)
        checkpoint_path = "keras_model.h5"

if resume_training and checkpoint_path != "":
    print('If resume_training is True while checkpoint_path is specified, resume_training will be set to False!')
    resume_training = False

# Retrieve last checkpoint
if resume_training:
    try:
        ckpt_dir_list = glob(full_model_path + '/ckpt/*')
        ckpt_dir_list.sort()
        last_ckpt_path = ckpt_dir_list[-1]
        print('Training will resume from checkpoint:', os.path.basename(last_ckpt_path))
    except IndexError:
        last_ckpt_path = None
        print('CheckpointError: No previous checkpoints were found, training from scratch.')
elif not resume_training and checkpoint_path != "":
    last_ckpt_path = checkpoint_path
    assert os.path.isfile(last_ckpt_path), 'checkpoint_path does not exist!'
else:
    last_ckpt_path = None

# Instantiate Unet3D
model = Unet3D(shape=training_shape)

# here we check that no model with the same name already exist
if not resume_training and os.path.exists(full_model_path):
    print(bcolors.WARNING + 'The model folder already exists and will be overwritten.' + bcolors.NORMAL)
    # print('!! WARNING: Folder already exists and will be overwritten !!')
    # shutil.rmtree(full_model_path)

# if not os.path.exists(full_model_path):
#     os.makedirs(full_model_path)

# Show sample image
if os.path.isdir(training_source):
    training_source_sample = sorted(glob(os.path.join(training_source, '*')))[0]
    training_target_sample = sorted(glob(os.path.join(training_target, '*')))[0]
else:
    training_source_sample = training_source
    training_target_sample = training_target

src_sample = tifffile.imread(training_source_sample)
src_sample = model._min_max_scaling(src_sample)
if binary_target:
    tgt_sample = tifffile.imread(training_target_sample).astype(np.bool)
else:
    tgt_sample = tifffile.imread(training_target_sample)

src_down = transform.downscale_local_mean(src_sample[0], (downscaling_in_xy, downscaling_in_xy))
tgt_down = transform.downscale_local_mean(tgt_sample[0], (downscaling_in_xy, downscaling_in_xy))

if random_crop:
    true_patch_size = None

    if src_down.shape[0] == training_shape[0]:
        x_rand = 0
    if src_down.shape[1] == training_shape[1]:
        y_rand = 0
    if src_down.shape[0] > training_shape[0]:
        x_rand = np.random.randint(src_down.shape[0] - training_shape[0])
    if src_down.shape[1] > training_shape[1]:
        y_rand = np.random.randint(src_down.shape[1] - training_shape[1])
    if src_down.shape[0] < training_shape[0] or src_down.shape[1] < training_shape[1]:
        raise ValueError('Patch shape larger than (downscaled) source shape')
else:
    true_patch_size = src_down.shape


def scroll_in_z(z):
    src_down = transform.downscale_local_mean(src_sample[z - 1], (downscaling_in_xy, downscaling_in_xy))
    tgt_down = transform.downscale_local_mean(tgt_sample[z - 1], (downscaling_in_xy, downscaling_in_xy))
    if random_crop:
        src_slice = src_down[x_rand:training_shape[0] + x_rand, y_rand:training_shape[1] + y_rand]
        tgt_slice = tgt_down[x_rand:training_shape[0] + x_rand, y_rand:training_shape[1] + y_rand]
    else:

        src_slice = transform.resize(src_down, (training_shape[0], training_shape[1]), mode='constant',
                                     preserve_range=True)
        tgt_slice = transform.resize(tgt_down, (training_shape[0], training_shape[1]), mode='constant',
                                     preserve_range=True)

    f = plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(src_slice, cmap='gray')
    plt.title('Training source (z = ' + str(z) + ')', fontsize=15)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(tgt_slice, cmap='magma')
    plt.title('Training target (z = ' + str(z) + ')', fontsize=15)
    plt.axis('off')
    plt.savefig(base_path + '/TrainingDataExample_Unet3D.png', bbox_inches='tight', pad_inches=0)
    # plt.close()


print('This is what the training images will look like with the chosen settings')
interact(scroll_in_z, z=widgets.IntSlider(min=1, max=src_sample.shape[0], step=1, value=0));
plt.show()
# Create a copy of an example slice and close the display.
scroll_in_z(z=int(src_sample.shape[0] / 2))
# If you close the display, then the users can't interactively inspect the data
# plt.close()

# Save model parameters
params = {'training_source': training_source,
          'training_target': training_target,
          'model_name': model_name,
          'model_path': model_path,
          'number_of_epochs': number_of_epochs,
          'batch_size': batch_size,
          'training_shape': training_shape,
          'downscaling': downscaling_in_xy,
          'true_patch_size': true_patch_size,
          'val_split': validation_split_in_percent / 100,
          'random_crop': random_crop}

params_df = pd.DataFrame.from_dict(params, orient='index')


model_spec = load_raw_resource_description(model_id)
if "keras_hdf5" not in model_spec.weights:
    print("Invalid bioimageio model")
else:
    url = model_spec.weights["keras_hdf5"].source
    r = requests.get(url, allow_redirects=True)
    open("keras_model.h5", 'wb').write(r.content)
    checkpoint_path = "keras_model.h5"


# Instantiate Unet3D
model = Unet3D(shape=training_shape)
