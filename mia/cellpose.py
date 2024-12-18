import itertools
from dataclasses import dataclass

from cellpose import transforms

from mia.hash import save_to_cache, compute_hash, load_from_cache


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

param_options = {
    "model_name": ["cyto3", "nuclei"], # ["cyto", "cyto2", "cyto3", "nuclei"]
    "channel_segment": [0], # [0, 1, 2, 3]
    "channel_nuclei": [0],
    "channel_axis": [None], # TODO
    "invert": [False], # [False, True]
    "normalize": [True], # [False, True]
    "diameter": [30, 50, 70, 100], # TODO good values (btw. None not working for 3D)
    "do_3D": [True], # TODO try False too
    "flow_threshold": [0.3, 0.5, 0.7], # [0.3, 0.4, 0.5, 0.6]
    "cellprob_threshold": [0.0, 0.1, 0.2, 0.5], # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    "interp": [False], # NOT AVAILABLE FOR 3D
    "min_size": [15], # TODO
    "max_size_fraction": [0.5], # TODO
    "niter": [100], # TODO
    "stitch_threshold": [0.0], # TODO
    "tile_overlap": [0.1], # TODO
    "type": ["Nuclei", "Membranes"],  # "Nuclei" or "Membranes"
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

# Generate all parameter combinations
evaluation_params = [
    EvaluationParams(**dict(zip(param_options.keys(), values)))
    for values in itertools.product(*param_options.values())
]



def evaluate_model(model, image, params, cache_dir, compute_masks=True):

    cache_key = compute_hash(image, params, compute_masks)

    cached_result = load_from_cache(cache_dir, cache_key)
    model_name = params.model_name

    if cached_result:
        print(f"\tLOADING FROM CACHE: {model_name}")
        masks, flows, styles, diams = cached_result
    else:
        print(f"\tEVALUATING: {model_name}")
        try:

            masks, flows, styles, diams = model.eval(
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
            save_to_cache(cache_dir, cache_key, masks, flows, styles, diams)

        except Exception as e:
            print(f"Error: {e}")
            masks = None
            flows = None
            styles = None
            diams = None

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

    return masks, flows, styles, diams
