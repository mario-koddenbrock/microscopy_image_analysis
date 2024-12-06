import datetime
import glob
import os

import cellpose
import napari
import numpy as np
import wandb
from cellpose import models, io
from cellpose.metrics import aggregated_jaccard_index, average_precision
from napari_animation import Animation

from mia.cellpose import evaluation_params
from mia.hash import compute_hash, load_from_cache, save_to_cache
from mia.results import ResultHandler
from mia.utils import check_set_gpu, check_paths

# Set the path to the ffmpeg executable - only needed for exporting animations
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
print(f"Cellpose version: {cellpose.version}")


def optimize_parameters(image_dir, output_dir, cache_dir="cache"):

    check_paths(image_dir, output_dir, cache_dir)

    # Do you want to show the viewer?
    show_viewer = False
    caching = False

    run_name = f"{os.path.basename(image_dir)}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    wandb.init(
        project="organoid_segmentation",
        name=run_name,
    )

    # get available torch device (CPU, GPU or MPS)
    device = check_set_gpu()

    # Load the Cellpose model
    model_dict = {
        "cyto": models.Cellpose(model_type='cyto', device=device),
        "cyto2": models.Cellpose(model_type='cyto2', device=device),
        "cyto3": models.Cellpose(model_type='cyto3', device=device),
        "nuclei": models.Cellpose(model_type='nuclei', device=device),
    }

    # Get the list of images
    image_paths = glob.glob(os.path.join(image_dir, "*.tif"))

    result_path = os.path.join(output_dir, "results.csv")
    if os.path.exists(result_path):
        os.remove(result_path)

    result_handler = ResultHandler(result_path)

    # Loop over the images
    for image_idx, image_path in enumerate(image_paths[1:]):

        image_name = os.path.basename(image_path).replace(".tif", "")
        ground_truth_path = image_path.replace("images", "Manual_meshes").replace(".tif", "-labels.tif")

        if os.path.exists(ground_truth_path):
            ground_truth = io.imread(ground_truth_path)
        else:
            print(f"Ground truth not found for {image_name}. Skipping.")
            continue

        # Read image with cellpose.io
        image = io.imread(image_path)

        for params in evaluation_params:

            if result_handler.is_result_present(params):
                continue  # Skip if result already exists

            cache_key = compute_hash(image, params)
            cached_result = load_from_cache(cache_dir, cache_key)

            print(f"Processing image {image_name} with parameters: {params}")

            try:
                if caching and cached_result:
                    print(f"Loaded cached result...")
                    masks, flows, styles, diams = cached_result
                else:
                    print(f"No cache found...")
                    model = model_dict[params["model_name"]]
                    masks, flows, styles, diams = model.eval(
                        image,
                        channels=[params["channel_segment"], params["channel_nuclei"]],
                        channel_axis=params["channel_axis"],
                        invert=params["invert"],
                        normalize=params["normalize"],
                        diameter=params["diameter"],
                        do_3D=params["do_3D"],
                        z_axis=0, # TODO: z-axis parameter
                    )
                    save_to_cache(cache_dir, cache_key, masks, flows, styles, diams)

                simple_jaccard = simple_iou(ground_truth, masks)
                # jaccard = jaccard_score(ground_truth, masks)
                # fscore = f1_score(ground_truth, masks)
                # precision, recall, fscore = boundary_scores(ground_truth, masks)

            except Exception as e:
                print(f"Error: {e}")
                continue

            # Log to W&B
            wandb.log({
                **params,
                "simple_jaccard": simple_jaccard,
                "image_name": image_name,
                "image_idx": image_idx,
            })

            result_handler.log_result(params, simple_jaccard, -1)

            if show_viewer:
                # Initialize the Napari viewer
                viewer = napari.Viewer()

                # Add the image to the viewer
                viewer.add_image(
                    image,
                    contrast_limits=[113, 1300],
                    name='Organoids',
                    colormap='gray',
                )

                # Add the labels to the viewer
                viewer.add_labels(
                    masks,
                    name=params["model_name"],
                    opacity=0.8,
                    blending='translucent',
                )

                # setting the viewer to the center of the image
                center = image.shape[0] // 2
                viewer.dims.set_point(0, center)

                napari.run()

        if image_idx > 10:
            break
    wandb.finish()



def view(image_dir, output_dir, cache_dir="cache"):

    check_paths(image_dir, output_dir, cache_dir)

    # Do you want to show the viewer or export the video?
    # TODO: they are mutually exclusive right now
    show_viewer = True
    export_video = False

    # List of channels, either of length 2 or of length number of images by 2.
    # First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
    # Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
    # For instance, to segment grayscale images, input [0,0].
    # To segment images with cells in green and nuclei in blue, input [2,3].
    # To segment one grayscale image and one image with cells in green and nuclei in blue, input [[0,0], [2,3]].
    # Defaults to [0,0].
    channels = [0, 0]

    # Load the Cellpose model
    model_dict = {
        "cyto": models.Cellpose(model_type='cyto', gpu=False),
        "cyto2": models.Cellpose(model_type='cyto2', gpu=False),
        "cyto3": models.Cellpose(model_type='cyto3', gpu=False),
        "nuclei": models.Cellpose(model_type='nuclei', gpu=False),
    }

    # Get the list of images
    image_paths = glob.glob(os.path.join(image_dir, "*.tif"))

    # Loop over the images
    for image_idx, image_path in enumerate(image_paths[1:]):

        image_name = os.path.basename(image_path).replace(".tif", "")

        # Read image with cellpose.io
        image = io.imread(image_path)

        ground_truth_path = image_path.replace("images", "Manual_meshes").replace(".tif", "-labels.tif")

        if os.path.exists(ground_truth_path):
            ground_truth = io.imread(ground_truth_path)
        else:
            print(f"Ground truth not found for {image_name}. Skipping.")
            ground_truth = None

        # # TODO: contrast normalization on image
        # image = rescale_intensity(image)

        print(f"image: {image_name}")
        print(f"shape: {image.shape}")
        print(f"dtype: {image.dtype}")
        print(f"range: ({np.min(image)}, {np.max(image)})")

        # iterate over all cellpose models
        for model_name, model in model_dict.items():
            print(f"Running model: {model_name}")

            params = {
                'model_name': model_name,
                'channel_segment': 0,
                'channel_nuclei': 0,
                'channel_axis': None,
                'invert': False,
                'normalize': False,
                'diameter': None,
                'do_3D': True,
            }

            cache_key = compute_hash(image, params)
            cached_result = load_from_cache(cache_dir, cache_key)

            if cached_result:
                print(f"Loaded cached result for {model_name}.")
                masks, flows, styles, diams = cached_result
            else:
                print(f"No cache found. Running eval for {model_name}.")
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

            # Initialize the Napari viewer
            viewer = napari.Viewer()

            if ground_truth is None:
                # Add the image to the viewer
                viewer.add_image(
                    image,
                    contrast_limits=[113, 1300],
                    name='Organoids',
                    colormap='gray',
                )


            if ground_truth is not None:
                simple_jaccard = simple_iou(ground_truth, masks)
                jaccard = jaccard_score(ground_truth, masks)
                fscore = f1_score(ground_truth, masks)
                # precision, recall, fscore = boundary_scores(ground_truth, masks)

                print(f"Simple Jaccard: {simple_jaccard}")
                print(f"Jaccard: {jaccard}")
                print(f"F1-Score: {fscore}")
                # print(f"Boundaries: {boundaries}")

                viewer.add_image(
                    ground_truth,
                    name="Ground truth",
                    # opacity=0.8,
                    # blending='translucent',
                    # colormap=matplotlib.cm.get_cmap('Set1'),
                )

            # Add the labels to the viewer
            viewer.add_labels(
                masks,
                name=model_name,
                opacity=0.8,
                blending='translucent',
                # colormap=matplotlib.cm.get_cmap('Set1'),
            )

            if export_video:
                # Create an animation object
                animation = Animation(viewer)

                # Loop over z-slices and capture frames
                for z in range(image.shape[0]):
                    viewer.dims.set_point(0, z)
                    animation.capture_keyframe()

                # Save the animation
                video_filename = f"{image_name}_{model_name}_animation.mp4"
                video_path = os.path.join(output_dir, video_filename)
                animation.animate(video_path, canvas_only=False)
                print(f"Saved animation to {video_path}")

                # Close the viewer to release resources
                viewer.close()

            if show_viewer:
                # setting the viewer to the center of the image
                center = image.shape[0] // 2
                viewer.dims.set_point(0, center)

                napari.run()

        if image_idx > 10:
            break


def jaccard_score(ground_truth, masks):
    aji_scores = aggregated_jaccard_index(ground_truth, masks)
    return np.mean(aji_scores)


def f1_score(ground_truth, masks):
    ap, tp, fp, fn = average_precision(ground_truth, masks)
    precision = np.sum(tp) / np.sum(tp + fp)
    recall = np.sum(tp) / np.sum(tp + fn)
    if precision + recall == 0:
        fscore = 0
    else:
        fscore = 2 * (precision * recall) / (precision + recall)
    return fscore


def simple_iou(ground_truth, masks):
    intersection = np.logical_and(ground_truth > 0, masks > 0).sum()
    union = np.logical_or(ground_truth > 0, masks > 0).sum()
    simple_jaccard = intersection / union if union > 0 else 0
    return simple_jaccard


if __name__ == "__main__":

    # directory containing the images
    image_dir = "Datasets/Organoids/20230712_P013T_cropped_isotropic/images"

    # directory to save the output
    output_dir = "Segmentation/Organoids/20230712_P013T_cropped_isotropic"

    # view(image_dir, output_dir)
    optimize_parameters(image_dir, output_dir)
