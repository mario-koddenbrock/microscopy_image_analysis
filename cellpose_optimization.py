import glob
import os
import random

import wandb
from mia.cellpose import (
    evaluate_model,
    EvaluationError,
    EvaluationParams,
    ensure_default_parameter,
)
from mia.results import ResultHandler
from mia.viz import show_napari, plot_eval


def optimize_parameters(
    param_options: dict,
    image_path: str = "",
    result_file: str = "",
    cache_dir: str = "cache",
    show_viewer: bool = False,
    log_wandb: bool = False,
):

    result_handler = ResultHandler(result_file, log_wandb)

    param_options = ensure_default_parameter(param_options)

    for model_name in param_options["model_name"]:
        for channel_segment in param_options["channel_segment"]:
            for channel_nuclei in param_options["channel_nuclei"]:
                for channel_axis in param_options["channel_axis"]:
                    for invert in param_options["invert"]:
                        for normalize in param_options["normalize"]:
                            for normalization_min in param_options["normalization_min"]:
                                for normalization_max in param_options[
                                    "normalization_max"
                                ]:
                                    for diameter in param_options["diameter"]:
                                        for do_3D in param_options["do_3D"]:
                                            for flow_threshold in param_options[
                                                "flow_threshold"
                                            ]:
                                                for cellprob_threshold in param_options[
                                                    "cellprob_threshold"
                                                ]:
                                                    for interp in param_options[
                                                        "interp"
                                                    ]:
                                                        for min_size in param_options[
                                                            "min_size"
                                                        ]:
                                                            for (
                                                                max_size_fraction
                                                            ) in param_options[
                                                                "max_size_fraction"
                                                            ]:
                                                                for (
                                                                    niter
                                                                ) in param_options[
                                                                    "niter"
                                                                ]:
                                                                    for (
                                                                        stitch_threshold
                                                                    ) in param_options[
                                                                        "stitch_threshold"
                                                                    ]:
                                                                        for (
                                                                            tile_overlap
                                                                        ) in param_options[
                                                                            "tile_overlap"
                                                                        ]:
                                                                            for (
                                                                                type
                                                                            ) in param_options[
                                                                                "type"
                                                                            ]:

                                                                                params = EvaluationParams(
                                                                                    model_name=model_name,
                                                                                    channel_segment=channel_segment,
                                                                                    channel_nuclei=channel_nuclei,
                                                                                    channel_axis=channel_axis,
                                                                                    invert=invert,
                                                                                    normalize=normalize,
                                                                                    normalization_min=normalization_min,
                                                                                    normalization_max=normalization_max,
                                                                                    diameter=diameter,
                                                                                    do_3D=do_3D,
                                                                                    flow_threshold=flow_threshold,
                                                                                    cellprob_threshold=cellprob_threshold,
                                                                                    interp=interp,
                                                                                    min_size=min_size,
                                                                                    max_size_fraction=max_size_fraction,
                                                                                    niter=niter,
                                                                                    stitch_threshold=stitch_threshold,
                                                                                    tile_overlap=tile_overlap,
                                                                                    type=type,
                                                                                )

                                                                                results = evaluate_model(
                                                                                    image_path,
                                                                                    params,
                                                                                    cache_dir,
                                                                                )

                                                                                if (
                                                                                    results
                                                                                    == EvaluationError.GROUND_TRUTH_NOT_AVAILABLE
                                                                                ):
                                                                                    break
                                                                                elif not isinstance(
                                                                                    results,
                                                                                    dict,
                                                                                ):
                                                                                    continue

                                                                                result_handler.log_result(
                                                                                    results,
                                                                                    params,
                                                                                )

                                                                                if show_viewer:
                                                                                    show_napari(
                                                                                        results,
                                                                                        params,
                                                                                    )

                                                                                if (
                                                                                    results[
                                                                                        "jaccard"
                                                                                    ]
                                                                                    > 0.95
                                                                                ):
                                                                                    print(
                                                                                        f"Found good parameters for {type} on {image_path}"
                                                                                    )
                                                                                    return

                                                                                if log_wandb:
                                                                                    wandb.finish()

        # options = {
        #     "model_name": ["cyto2_cp3", "cyto", "cyto2", "cyto3", "nuclei", "tissuenet_cp3", "livecell_cp3", "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", "deepbacs_cp3"],
        #     "channel_segment": [0],  # [0, 1, 2, 3]
        #     "channel_nuclei": [0],
        #     "channel_axis": [None],  # TODO
        #     "invert": [False],  # Dont do this
        #     "normalize": [True],  # Always do this
        #     "normalization_min": [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        #     "normalization_max": [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 99.5],
        #     "diameter": [10, 17, 30, 50, 70, 100],  # TODO good values (btw. None not working for 3D)
        #     "do_3D": [True],  # TODO try False too
        #     "flow_threshold": [0.1, 0.3, 0.5, 0.7],  # [0.3, 0.4, 0.5, 0.6]
        #     "cellprob_threshold": [0.0, 0.1, 0.2, 0.5],  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        #     "interp": [False],  # NOT AVAILABLE FOR 3D
        #     "min_size": [15, 25, 50],  # TODO
        #     "max_size_fraction": [0.5],  # TODO
        #     "niter": [100],  # TODO
        #     "stitch_threshold": [0.0],  # TODO
        #     "tile_overlap": [0.1],  # TODO
        #     "type": ["Nuclei", "Membranes"],
        # }


if __name__ == "__main__":

    # Set the random seed based on the image index
    random.seed(42)
    log_wandb = False
    main_folder = "Datasets/P013T/"

    # get all the subfolder
    subfolders = glob.glob(os.path.join(main_folder, "*"))

    # Combine all image paths into one list
    image_paths = [
        image_path
        for folder in subfolders
        if os.path.isdir(folder)
        for image_path in glob.glob(
            os.path.join(folder, "images_cropped_isotropic", "*.tif")
        )
    ]

    image_paths = [image_paths[5]]

    # get all the subfolder
    subfolders = glob.glob(os.path.join(main_folder, "*"))
    result_file_models = os.path.join(main_folder, "results_models.csv")
    result_file_normalize = os.path.join(main_folder, "results_normalize.csv")

    for image_idx, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path).replace(".tif", "")

        if log_wandb:
            wandb.init(project="organoid_segmentation", name=f"{image_name}")

        options_model = {
            "model_name": [
                "cyto2_cp3",
                "cyto",
                "cyto2",
                "cyto3",
                "nuclei",
                "tissuenet_cp3",
                "livecell_cp3",
                "yeast_PhC_cp3",
                "yeast_BF_cp3",
                "bact_phase_cp3",
                "bact_fluor_cp3",
                "deepbacs_cp3",
            ]
        }
        optimize_parameters(options_model, image_path, result_file_models)

        options_normalization = {"normalization_min": [0.5, 1, 2, 3, 5, 7, 10]}
        optimize_parameters(options_normalization, image_path, result_file_normalize)

        options_normalization = {"normalization_max": [90, 93, 95, 97, 98, 99, 99.5]}
        optimize_parameters(options_normalization, image_path, result_file_normalize)

    plot_eval(options_model)
    plot_eval(options_normalization)
