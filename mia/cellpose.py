import itertools

# MODEL EVALUATION PARAMETERS
# x (list or array): List or array of images. Can be list of 2D/3D images, or array of 2D/3D images, or 4D image array.
# batch_size (int, optional): Number of 224x224 patches to run simultaneously on the GPU. Can make smaller or bigger depending on GPU memory usage. Defaults to 8.
# channels (list, optional): List of channels, either of length 2 or of length number of images by 2. First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue). Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue). For instance, to segment grayscale images, input [0,0]. To segment images with cells in green and nuclei in blue, input [2,3]. To segment one grayscale image and one image with cells in green and nuclei in blue, input [[0,0], [2,3]]. Defaults to [0,0].
# channel_axis (int, optional): If None, channels dimension is attempted to be automatically determined. Defaults to None.
# invert (bool, optional): Invert image pixel intensity before running network (if True, image is also normalized). Defaults to False.
# normalize (bool, optional): If True, normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel; can also pass dictionary of parameters (see CellposeModel for details). Defaults to True.
# diameter (float, optional): If set to None, then diameter is automatically estimated if size model is loaded. Defaults to 30..
# do_3D (bool, optional): Set to True to run 3D segmentation on 4D image input. Defaults to False.

channel_segment_list = [0, 1, 2, 3]
channel_nuclei_list = [0, 1, 2, 3]
channel_axis_list = [None] # TODO
invert_list = [False, True]
normalize_list = [False, True] # TODO iterate over normalization parameters
diameter_list = [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100] # TODO
do_3D_list = [True] # TODO try False too
model_list = ["cyto", "cyto2", "cyto3", "nuclei"]



# getting dict of all possible parameters
evaluation_params = [
    {
        "model_name": m,
        "channel_segment": cs,
        "channel_nuclei": cn,
        "channel_axis": ca,
        "invert": inv,
        "normalize": norm,
        "diameter": dia,
        "do_3D": d3
    }
    for m, cs, cn, ca, inv, norm, dia, d3 in itertools.product(
        model_list, channel_segment_list, channel_nuclei_list, channel_axis_list, invert_list, normalize_list, diameter_list, do_3D_list
    )
]
