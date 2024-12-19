import itertools
import os
import time
from medpy.metric.binary import jc

from dataclasses import dataclass
from enum import Enum

import wandb
import yaml
import numpy as np
from cellpose import transforms
from cellpose.metrics import aggregated_jaccard_index
from cellpose.models import CellposeModel
from skimage.metrics import adapted_rand_error
from sklearn.metrics import jaccard_score

from mia.file_io import load_image_with_gt
from mia.hash import save_to_cache, compute_hash, load_from_cache
from mia.utils import check_set_gpu

# The dataset-specific models were trained on the training images from the following datasets:
# tissuenet_cp3: tissuenet dataset.
# livecell_cp3: livecell dataset
# yeast_PhC_cp3: YEAZ dataset
# yeast_BF_cp3: YEAZ dataset
# bact_phase_cp3: omnipose dataset
# bact_fluor_cp3: omnipose dataset
# deepbacs_cp3: deepbacs dataset
# cyto2_cp3: cellpose dataset

# We have a nuclei model and a super-generalist cyto3 model.
# There are also two older models, cyto, which is trained on only the Cellpose training set, and cyto2, which is also trained on user-submitted images.

# nchan (int, optional): Number of channels to use as input to network, default is 2 (cyto + nuclei) or (nuclei + zeros). TODO
# backbone_list # TODO
# anisotropic_list # TODO

model_list = ["cyto", "cyto2", "cyto3", "nuclei", "tissuenet_cp3", "livecell_cp3", "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", "deepbacs_cp3", "cyto2_cp3"]

param_options = {
    "model_name": model_list,
    "channel_segment": [0], # [0, 1, 2, 3]
    "channel_nuclei": [0],
    "channel_axis": [None], # TODO
    "invert": [False, True], # [False, True]
    "normalize": [False, True], # [False, True]
    "diameter": [10, 17, 30, 50, 70, 100], # TODO good values (btw. None not working for 3D)
    "do_3D": [True], # TODO try False too
    "flow_threshold": [0.1, 0.3, 0.5, 0.7], # [0.3, 0.4, 0.5, 0.6]
    "cellprob_threshold": [0.0, 0.1, 0.2, 0.5], # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    "interp": [False], # NOT AVAILABLE FOR 3D
    "min_size": [15], # TODO
    "max_size_fraction": [0.5], # TODO
    "niter": [100], # TODO
    "stitch_threshold": [0.0], # TODO
    "tile_overlap": [0.1], # TODO
    "type": [""],  # just a placeholder for "Nuclei" or "Membranes"
}

# Define a dataclass to store the evaluation parameters
@dataclass
class EvaluationParams:
    model_name: str
    channel_segment: int
    channel_nuclei: int
    channel_axis: any
    invert: bool
    normalize: bool
    diameter: int
    do_3D: bool
    flow_threshold: float
    cellprob_threshold: float
    interp: bool
    min_size: int
    max_size_fraction: float
    niter: int
    stitch_threshold: float
    tile_overlap: float
    type: str

    def to_yaml(self, yaml_file: str):
        """
        Save an EvaluationParams object to a YAML file.

        Parameters:
            self (EvaluationParams): The object containing evaluation parameters.
            yaml_file (str): The path to save the YAML file.
        """
        config = {
            "model_name": self.model_name,
            "channel_segment": self.channel_segment,
            "channel_nuclei": self.channel_nuclei,
            "channel_axis": self.channel_axis,
            "invert": self.invert,
            "normalize": self.normalize,
            "diameter": self.diameter,
            "do_3D": self.do_3D,
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "interp": self.interp,
            "min_size": self.min_size,
            "max_size_fraction": self.max_size_fraction,
            "niter": self.niter,
            "stitch_threshold": self.stitch_threshold,
            "tile_overlap": self.tile_overlap,
            "type": self.type
        }
        try:
            with open(yaml_file, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            print(f"Saved EvaluationParams to {yaml_file}")
        except Exception as e:
            print(f"Error saving EvaluationParams to YAML: {e}")

# Generate all parameter combinations
evaluation_params = [
    EvaluationParams(**dict(zip(param_options.keys(), values)))
    for values in itertools.product(*param_options.values())
]




def evaluate_model(image_path, params, cache_dir="cache", compute_masks=True):
    t0 = time.time()

    image_name = os.path.basename(image_path).replace(".tif", "")
    device = check_set_gpu()  # get available torch device (CPU, GPU or MPS)
    ground_truth, image_orig = load_image_with_gt(image_path, params.type)

    if ground_truth is None:
        return EvaluationError.GROUND_TRUTH_NOT_AVAILABLE

    print(f"Processing {image_name} ({device})")

    if params.type == "Nuclei":
        channel_idx = 0
    elif params.type == "Membranes":
        channel_idx = 1
    else:
        raise ValueError(f"Invalid type: {params.type}")

    # Get the right channel
    image = image_orig[:, channel_idx, :, :] if image_orig.ndim == 4 else image_orig


    # # TODO: contrast normalization on image
    # image = rescale_intensity(image)

    # plot the intensity distribution of the image
    # plot_intensity(image)

    cache_key = compute_hash(image, params, compute_masks)

    cached_result = load_from_cache(cache_dir, cache_key)
    model_name = params.model_name

    if cached_result:
        print(f"\tLOADING FROM CACHE: {model_name}")
        masks, flows, styles, diams = cached_result
    else:
        print(f"\tEVALUATING: {model_name}")
        try:

            # # Load the Cellpose model
            # model = Cellpose(
            #     model_type = params.model_name,
            #     device = device,
            #     nchan = 2,  # TODO check if this is correct = 1?
            #     # diam_mean = 30,
            #     gpu = False,
            # )


            # diam_mean = 30.  # default for any cyto model
            # nuclear = "nuclei" in params.model_name
            # if nuclear:
            #     diam_mean = 17.

            model = CellposeModel(device=device, gpu=False, model_type=params.model_name,
                                    diam_mean=params.diameter, nchan=2,
                                    backbone="default")

            masks, flows, styles = model.eval(
                image,
                cellprob_threshold=params.cellprob_threshold,
                channel_axis=params.channel_axis,
                channels=[params.channel_segment, params.channel_nuclei],
                compute_masks=compute_masks,
                diameter=params.diameter,
                do_3D=params.do_3D,
                flow_threshold=params.flow_threshold,
                invert=params.invert,
                max_size_fraction=0.5, # default
                min_size=15, # default
                normalize=params.normalize,
                z_axis=0,  # TODO: z-axis parameter always 0?
            )
            save_to_cache(cache_dir, cache_key, masks, flows, styles, params.diameter)

        except Exception as e:
            print(f"Error: {e}")
            return EvaluationError.EVALUATION_ERROR

    if not compute_masks:
        # dP_colors = flows[0]
        dP = flows[1]
        cellprob = flows[2]
        nchan = 2  # TODO
        x = transforms.convert_image(
            image,
            [params.channel_segment, params.channel_nuclei],
            channel_axis=params.channel_axis,
            z_axis=0,
            do_3D=(params.do_3D or params.stitch_threshold > 0),
            nchan=nchan)

        masks = model.cp._compute_masks(
            x.shape, dP, cellprob,
            flow_threshold=params.flow_threshold,
            cellprob_threshold=params.cellprob_threshold,
            interp=params.interp,
            min_size=params.min_size,
            max_size_fraction=params.max_size_fraction,
            niter=params.niter,
            stitch_threshold=params.stitch_threshold,
            do_3D=params.do_3D,
        )

        masks = masks.squeeze()

    if masks is None:
        print(f"Error: No masks found with parameters")
        return EvaluationError.EMPTY_MASKS

    else:
        # jaccard = jaccard_score(ground_truth, masks)
        # simple_jaccard = simple_iou(ground_truth, masks)
        # iout, preds = mask_ious(ground_truth, masks)
        # jaccard = jaccard_score(ground_truth, masks)
        # fscore = f1_score(ground_truth, masks)

        # jaccard = jc(masks, ground_truth)

        jaccard_sklearn = jaccard_score(ground_truth.flatten(), masks.flatten(), average='micro')
        aji_scores = aggregated_jaccard_index(ground_truth, masks)
        jaccard_cellpose = np.mean(aji_scores)

        are, precision, recall = adapted_rand_error(ground_truth, masks)
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        print(f"\tAdapted Rand Error: {are:.2f}")
        print(f"\tPrecision: {precision:.2f}")
        print(f"\tRecall: {recall:.2f}")
        print(f"\tF1: {f1:.2f}")
        print(f"\tJaccard (sklearn): {jaccard_sklearn:.2f}")
        print(f"\tJaccard (cellpose): {jaccard_cellpose:.2f}")

    # fig = plt.figure(figsize=(12, 5))
    # plot.show_segmentation(fig, image, masks, flows, channels=[0, 0])

    duration = time.time() - t0

    results = {
        "image": image,
        "image_name": image_name,
        "ground_truth": ground_truth,
        "masks": masks,
        "are": are,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard_sklearn": jaccard_sklearn,
        "jaccard_cellpose": jaccard_cellpose,
        "duration": duration,
    }





    return results


def read_yaml(yaml_file: str = "") -> EvaluationParams:
    """
    Read a YAML file containing model configuration and return an EvaluationParams object.

    Parameters:
        yaml_file (str): Path to the YAML configuration file.

    Returns:
        EvaluationParams: An object populated with the parameters from the YAML file.
    """
    # Load configuration from YAML
    try:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File '{yaml_file}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")
        return None

    print(f"Loaded configuration from {yaml_file}:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Map configuration to EvaluationParams
    params = EvaluationParams(
        model_name=config.get('model_name'),
        channel_segment=config.get('channel_segment'),
        channel_nuclei=config.get('channel_nuclei'),
        channel_axis=config.get('channel_axis'),
        invert=config.get('invert'),
        normalize=config.get('normalize'),
        diameter=config.get('diameter'),
        do_3D=config.get('do_3D'),
        flow_threshold=config.get('flow_threshold'),
        cellprob_threshold=config.get('cellprob_threshold'),
        interp=config.get('interp'),
        min_size=config.get('min_size'),
        max_size_fraction=config.get('max_size_fraction'),
        niter=config.get('niter'),
        stitch_threshold=config.get('stitch_threshold'),
        tile_overlap=config.get('tile_overlap'),
        type=config.get('type')
    )

    return params


class EvaluationError(Enum):
    GROUND_TRUTH_NOT_AVAILABLE = "Ground truth not available"
    EVALUATION_ERROR = "Evaluation error"
    EMPTY_MASKS = "Empty masks"
