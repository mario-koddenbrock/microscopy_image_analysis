import itertools

from mia.hash import save_to_cache, compute_hash, load_from_cache

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
channel_nuclei_list = [0]
channel_axis_list = [0] # TODO
invert_list = [False, True]
normalize_list = [False, True] # TODO iterate over normalization parameters
diameter_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100] # TODO good values (btw. None not working for 3D)
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


def evaluate_model(model, image, params, cache_dir):
    cache_key = compute_hash(image, params)
    cached_result = load_from_cache(cache_dir, cache_key)
    model_name = params["model_name"]

    if cached_result:
        print(f"Loaded cached result for {model_name}.")
        masks, flows, styles, diams = cached_result
    else:
        print(f"No cache found. Running eval for {model_name}.")
        try:
            masks, flows, styles, diams = model.eval(
                image,
                channels=[params["channel_segment"], params["channel_nuclei"]],
                channel_axis=params["channel_axis"],
                invert=params["invert"],
                normalize=params["normalize"],
                diameter=params["diameter"],
                do_3D=params["do_3D"],
                z_axis=0,  # TODO: z-axis parameter
            )
            save_to_cache(cache_dir, cache_key, masks, flows, styles, diams)

        except Exception as e:
            print(f"Error: {e}")
            masks = None
            flows = None
            styles = None
            diams = None

        save_to_cache(cache_dir, cache_key, masks, flows, styles, diams)

    return masks, flows, styles, diams
