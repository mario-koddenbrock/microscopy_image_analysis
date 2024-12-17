import glob
import os

import napari

from cellpose import io
from cellpose.metrics import boundary_scores, mask_ious
from cellpose.models import Cellpose
from segmentation_models_pytorch.utils.functional import jaccard
from skimage.metrics import adapted_rand_error
from tqdm import tqdm

import wandb
from mia.cellpose import evaluation_params, evaluate_model
from mia.file_io import get_cellpose_ground_truth
from mia.hash import compute_hash
from mia.metrics import simple_iou, jaccard_score, f1_score
from mia.results import ResultHandler
from mia.utils import check_paths, check_set_gpu
from mia.viz import plot_intensity, show_napari


# TODO use CellposeModel instead of Cellpose class

# setting PYTORCH_ENABLE_MPS_FALLBACK=1


def optimize_parameters(
        image_dir: str = "",
        output_dir: str = "",
        cache_dir: str = "cache",
        type:str = "Nuclei",  # "Nuclei" or "Membranes"
        log_wandb: bool = False,
        show_viewer: bool = False,
):

    device = check_set_gpu() # get available torch device (CPU, GPU or MPS)
    result_path = check_paths(image_dir, output_dir, cache_dir)
    image_paths = glob.glob(os.path.join(image_dir, "*.tif")) # Get the list of images
    result_handler = ResultHandler(result_path)

    # Loop over the images
    for image_idx, image_path in enumerate(image_paths):

        image_name = os.path.basename(image_path).replace(".tif", "")

        # Read image with cellpose.io
        image_orig = io.imread(image_path)
        ground_truth = get_cellpose_ground_truth(image_path, image_name)


        for param_idx, params in enumerate(evaluation_params):

            type = params["type"]
            if type == "Nuclei":
                channel_idx = 0
            elif type == "Membranes":
                channel_idx = 1
            else:
                raise ValueError(f"Invalid type: {type}")

            # Get the right channel
            image = image_orig[:, channel_idx, :, :] if image_orig.ndim == 4 else image_orig

            # # TODO: contrast normalization on image
            # image = rescale_intensity(image)

            # plot the intensity distribution of the image
            # plot_intensity(image)

            cache_key = compute_hash(image, params)
            if log_wandb:
                wandb.init(
                    project="organoid_segmentation",
                    name=cache_key,
                    config=params,
                )

            # Load the Cellpose model
            model = Cellpose(
                model_type = params["model_name"],
                device = device,
                nchan = 2,  # TODO check if this is correct = 1?
                # diam_mean = 30, # TODO how to change this?
                gpu = False,
            )

            print(f"{image_idx+1}/{len(image_paths)}: Processing image {image_name} {image.shape}")
            print(f"\t{param_idx+1}/{len(evaluation_params)}: Parameter:")
            # for k, v in params.items():
            #     print(f"\t\t{k}: {v}")

            masks, flows, styles, diams = evaluate_model(model, image, params, cache_dir)

            if masks is None:
                print(f"Error: No masks found for {image_name} with parameters: {params}")
                continue

            else:
                # jaccard = jaccard_score(ground_truth, masks)
                # simple_jaccard = simple_iou(ground_truth, masks)
                # iout, preds = mask_ious(ground_truth, masks)
                # jaccard = jaccard_score(ground_truth, masks)
                # fscore = f1_score(ground_truth, masks)

                are, precision, recall = adapted_rand_error(ground_truth, masks)
                f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                print(f"\tAdapted Rand Error: {are:.2f}")
                print(f"\tPrecision: {precision:.2f}")
                print(f"\tRecall: {recall:.2f}")
                print(f"\tF1: {f1:.2f}")

            # fig = plt.figure(figsize=(12, 5))
            # plot.show_segmentation(fig, image, masks, flows, channels=[0, 0])

            # Log to W&B
            if log_wandb:
                wandb.log({
                    **params,
                    "adapted_rand_error": are,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "image_name": image_name,
                    "image_idx": image_idx,
                })
                wandb.finish()

            result_handler.log_result(params, are, precision, recall, f1)

            if show_viewer:
                show_napari(image, ground_truth, masks, params)



if __name__ == "__main__":

    main_folder = "Datasets/P013T/"

    # get all the subfolder
    subfolders = glob.glob(os.path.join(main_folder, "*"))

    for folder in subfolders[1:]:

        if not os.path.isdir(folder):
            continue

        # directory containing the images
        image_dir = os.path.join(folder, "images_cropped_isotropic")

        # directory to save the output
        output_dir = image_dir.replace("Datasets", "Segmentation")

        optimize_parameters(image_dir, output_dir)
