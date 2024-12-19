import glob
import os
import random

import wandb

from mia.cellpose import evaluation_params, evaluate_model, EvaluationError
from mia.results import ResultHandler
from mia.utils import check_paths
from mia.viz import show_napari

def optimize_parameters(
        image_dir: str = "",
        output_dir: str = "",
        result_file: str = "",
        cache_dir: str = "cache",
        show_viewer: bool = False,
        num_parameters: int = 100,
        log_wandb: bool = True,
):

    check_paths(image_dir, output_dir, cache_dir)
    image_paths = glob.glob(os.path.join(image_dir, "*.tif")) # Get the list of images

    result_handler = ResultHandler(result_file)

    # Loop over the images
    for image_idx, image_path in enumerate(image_paths):

        image_name = os.path.basename(image_path).replace(".tif", "")

        # Set the random seed based on the image index
        random.seed(image_idx)

        # get indices of the parameters to evaluate
        idx = random.sample(range(len(evaluation_params)), num_parameters)

        print(f"Image {image_idx+1}/{len(image_paths)}: {image_path}")
        sampled_params = [evaluation_params[i] for i in idx]

        for type in ["Nuclei", "Membranes"]:

            if log_wandb:
                wandb.init(
                    project="organoid_segmentation",
                    name=f"{image_name}_{type}",
                )

            for param_idx, params in enumerate(sampled_params):

                params.type = type
                print(f"Parameter set {param_idx+1}/{num_parameters}")
                results = evaluate_model(image_path, params, cache_dir, log_wandb=log_wandb)

                if results == EvaluationError.GROUND_TRUTH_NOT_AVAILABLE:
                    break
                elif not isinstance(results, dict):
                    continue

                result_handler.log_result(results, params)

                if show_viewer:
                    show_napari(results, params)

                if results["f1"] > 0.95:
                    print(f"Found good parameters for {type} on {image_path}")
                    break




if __name__ == "__main__":

    main_folder = "Datasets/P013T/"

    # get all the subfolder
    subfolders = glob.glob(os.path.join(main_folder, "*"))
    result_file = os.path.join(main_folder, "results.csv")

    for folder in subfolders:

        if not os.path.isdir(folder):
            continue

        # directory containing the images
        image_dir = os.path.join(folder, "images_cropped_isotropic")

        # directory to save the output
        output_dir = image_dir.replace("Datasets", "Segmentation")

        optimize_parameters(image_dir, output_dir, result_file)
