import itertools

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

model_list = ["cyto3", "nuclei"] # ["cyto", "cyto2", "cyto3", "nuclei"]


cellprob_threshold_list = [0.0, 0.1, 0.2] # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
channel_axis_list = [None] # TODO
channel_nuclei_list = [0]
channel_segment_list = [0] # [0, 1, 2, 3]
diameter_list = [30, 50, 70] # TODO good values (btw. None not working for 3D)
do_3D_list = [True] # TODO try False too
flow_threshold_list = [0.3, 0.5] # [0.3, 0.4, 0.5, 0.6]
interp_list = [False] # NOT AVAILABLE FOR 3D
invert_list = [False] # [False, True]
max_size_fraction_list = [0.5] # TODO
min_size_list = [15] # TODO
niter_list = [100] # TODO
normalize_list = [True] # [False, True]
stitch_threshold_list = [0.0] # TODO
tile_overlap_list = [0.1] # TODO
type_list = ["Nuclei", "Membranes"]  # "Nuclei" or "Membranes"

# nchan (int, optional): Number of channels to use as input to network, default is 2 (cyto + nuclei) or (nuclei + zeros). TODO
# backbone_list # TODO
# anisotropic_list # TODO

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
        "do_3D": d3,
        "flow_threshold": ft,
        "cellprob_threshold": ct,
        "interp": interp,
        "min_size": ms,
        "max_size_fraction": msf,
        "niter": niter,
        "stitch_threshold": st,
        "tile_overlap": to,
        "type": t,
    }
    for m, cs, cn, ca, inv, norm, dia, d3, ft, ct, interp, ms, msf, niter, st, to, t in itertools.product(
        model_list,
        channel_segment_list,
        channel_nuclei_list,
        channel_axis_list,
        invert_list,
        normalize_list,
        diameter_list,
        do_3D_list,
        flow_threshold_list,
        cellprob_threshold_list,
        interp_list,
        min_size_list,
        max_size_fraction_list,
        niter_list,
        stitch_threshold_list,
        tile_overlap_list,
        type_list,
    )
]



def evaluate_model(model, image, params, cache_dir, compute_masks=True):

    cache_key = compute_hash(image, params, compute_masks)

    cached_result = load_from_cache(cache_dir, cache_key)
    model_name = params["model_name"]

    if cached_result:
        print(f"\tLOADING FROM CACHE: {model_name}")
        masks, flows, styles, diams = cached_result
    else:
        print(f"\tEVALUATING: {model_name}")
        try:

            masks, flows, styles, diams = model.eval(
                image,
                cellprob_threshold=params["cellprob_threshold"],
                channel_axis=params["channel_axis"],
                channels=[params["channel_segment"], params["channel_nuclei"]],
                compute_masks=compute_masks,
                diameter=params["diameter"],
                do_3D=params["do_3D"],
                flow_threshold=params["flow_threshold"],
                invert=params["invert"],
                max_size_fraction=0.5, # default
                min_size=15, # default
                normalize=params["normalize"],
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
            [params["channel_segment"], params["channel_nuclei"]],
            channel_axis=params["channel_axis"],
            z_axis=0,
            do_3D=(params["do_3D"] or params["stitch_threshold"] > 0),
            nchan=nchan)

        masks = model.cp._compute_masks(
            x.shape, dP, cellprob,
            flow_threshold=params["flow_threshold"],
            cellprob_threshold=params["cellprob_threshold"],
            interp=params["interp"],
            min_size=params["min_size"],
            max_size_fraction=params["max_size_fraction"],
            niter=params["niter"],
            stitch_threshold=params["stitch_threshold"],
            do_3D=params["do_3D"],
        )

        masks = masks.squeeze()

    return masks, flows, styles, diams
