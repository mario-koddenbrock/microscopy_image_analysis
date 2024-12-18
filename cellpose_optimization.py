import glob
import os
import random

from cellpose import io

from mia.cellpose import evaluation_params, evaluate_model
from mia.file_io import get_cellpose_ground_truth
from mia.results import ResultHandler
from mia.utils import check_paths
from mia.viz import show_napari


# TODO use CellposeModel instead of Cellpose class
# setting PYTORCH_ENABLE_MPS_FALLBACK=1


def optimize_parameters(
        image_dir: str = "",
        output_dir: str = "",
        result_file: str = "",
        cache_dir: str = "cache",
        show_viewer: bool = False,
        num_parameters: int = 100,
):

    check_paths(image_dir, output_dir, cache_dir)
    image_paths = glob.glob(os.path.join(image_dir, "*.tif")) # Get the list of images

    result_handler = ResultHandler(result_file)

    # Loop over the images
    for image_idx, image_path in enumerate(image_paths):

        # Set the random seed based on the image index
        random.seed(image_idx)

        # Sample a fixed number of parameter combinations
        sampled_params = random.sample(evaluation_params, num_parameters)
        for param_idx, params in enumerate(sampled_params):

            results = evaluate_model(image_path, params, cache_dir)

            if results is not None:
                result_handler.log_result(results, params)

            if show_viewer:
                show_napari(results, params)




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
