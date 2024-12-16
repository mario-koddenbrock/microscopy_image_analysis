import glob
import os

import napari
from cellpose import models, io

import wandb
from mia.cellpose import evaluation_params, evaluate_model
from mia.file_io import get_cellpose_ground_truth
from mia.hash import compute_hash
from mia.metrics import simple_iou
from mia.results import ResultHandler
from mia.utils import check_set_gpu, check_paths


def optimize_parameters(
        image_dir: str = "",
        output_dir: str = "",
        cache_dir: str = "cache",
        type:str = "Nuclei",  # "Nuclei" or "Membranes"
        log_wandb: bool = False,
):

    check_paths(image_dir, output_dir, cache_dir)

    # Do you want to show the viewer?
    show_viewer = False

    # get available torch device (CPU, GPU or MPS)
    device = check_set_gpu()

    # Load the Cellpose model
    model_args = {
        "device": device,
        "nchan": 2, # TODO check if this is correct = 1?
        # "diam_mean": 30, # TODO how to change this?
        "gpu": False,
    }

    model_dict = {
        "cyto": models.Cellpose(model_type='cyto', **model_args),
        "cyto2": models.Cellpose(model_type='cyto2', **model_args),
        "cyto3": models.Cellpose(model_type='cyto3', **model_args),
        "nuclei": models.Cellpose(model_type='nuclei', **model_args),
    }

    # Get the list of images
    image_paths = glob.glob(os.path.join(image_dir, "*.tif"))

    result_path = os.path.join(output_dir, "results.csv")
    if os.path.exists(result_path):
        os.remove(result_path)

    if type == "Nuclei":
        channel_idx = 0
    elif type == "Membranes":
        channel_idx = 1
    else:
        raise ValueError(f"Invalid type: {type}")


    result_handler = ResultHandler(result_path)

    # Loop over the images
    for image_idx, image_path in enumerate(image_paths[1:]):

        image_name = os.path.basename(image_path).replace(".tif", "")

        # Read image with cellpose.io
        image = io.imread(image_path)
        ground_truth = get_cellpose_ground_truth(image_path, image_name)

        # # TODO: contrast normalization on image
        # image = rescale_intensity(image)

        if image.ndim == 4:
            image = image[:, channel_idx, :, :]

        # plot the intensety distribution of the image
        # plot_intensety(image)

        for params in evaluation_params:

            # if result_handler.is_result_present(params):
            #     continue  # Skip if result already exists

            cache_key = compute_hash(image, params)
            if log_wandb:
                wandb.init(
                    project="organoid_segmentation",
                    name=cache_key,
                    config=params,
                )

            model = model_dict[params["model_name"]]
            print(f"Processing image {image_name} ({image.shape}) with parameters: {params}")

            masks, flows, styles, diams = evaluate_model(model, image, params, cache_dir)

            simple_jaccard = simple_iou(ground_truth, masks)
            # jaccard = jaccard_score(ground_truth, masks)
            # fscore = f1_score(ground_truth, masks)
            # precision, recall, fscore = boundary_scores(ground_truth, masks)

            if masks is None:
                print(f"Error: No masks found for {image_name} with parameters: {params}")
                continue

            print(f"Simple Jaccard: {simple_jaccard}")

            # fig = plt.figure(figsize=(12, 5))
            # plot.show_segmentation(fig, image, masks, flows, channels=[0, 0])

            # Log to W&B
            if log_wandb:
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

    if log_wandb:
        wandb.finish()





if __name__ == "__main__":

    main_folder = "Datasets/P013T/"

    # get all the subfolder
    subfolders = glob.glob(os.path.join(main_folder, "*"))

    for folder in subfolders:

        if not os.path.isdir(folder):
            continue

        # directory containing the images
        image_dir = os.path.join(folder, "images_cropped_isotropic")

        # directory to save the output
        output_dir = image_dir.replace("Datasets", "Segmentation")

        optimize_parameters(image_dir, output_dir)
