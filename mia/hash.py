import hashlib
import json
import os
from dataclasses import asdict

import numpy as np


def filter_model_parameter(params):
    model_keys = [
        "model_name",
        "channel_segment",
        "channel_nuclei",
        "channel_axis",
        "diameter",
        "do_3D",
        "stitch_threshold",
        "normalize",
        "invert",
        "tile_overlap",
    ]
    return {k: params.get(k, None) for k in model_keys}


def compute_hash(image, parameters, compute_masks: bool = True):
    """
    Compute a unique hash for the image and parameters.
    """
    if not compute_masks:
        # TODO adapt to new parameter object
        parameters = filter_model_parameter(parameters)

    image_hash = hashlib.sha256(image.tobytes()).hexdigest()
    param_hash = hashlib.sha256(
        json.dumps(asdict(parameters), sort_keys=True).encode()
    ).hexdigest()
    return f"{image_hash}_{param_hash}"


def save_to_cache(cache_dir, cache_key, masks, flows, styles, diams):
    """
    Save data to the cache directory with separate files for each component.
    """
    cache_key_dir = os.path.join(cache_dir, cache_key)
    if not os.path.exists(cache_key_dir):
        os.makedirs(cache_key_dir)

    # Save each component separately
    np.save(os.path.join(cache_key_dir, "masks.npy"), masks.astype(np.uint16))
    np.save(os.path.join(cache_key_dir, "styles.npy"), styles.astype(np.float32))
    np.save(
        os.path.join(cache_key_dir, "diams.npy"), np.array([diams], dtype=np.float32)
    )

    # Save flows as separate files
    flows_dir = os.path.join(cache_key_dir, "flows")
    if not os.path.exists(flows_dir):
        os.makedirs(flows_dir)
    for i, flow in enumerate(flows):
        np.save(os.path.join(flows_dir, f"flow_{i}.npy"), flow.astype(np.uint8))


def load_from_cache(cache_dir, cache_key):
    """
    Load data from the cache directory with separate files for each component.
    """
    cache_key_dir = os.path.join(cache_dir, cache_key)
    if not os.path.exists(cache_key_dir):
        return None

    # Load each component
    masks = np.load(os.path.join(cache_key_dir, "masks.npy"))
    styles = np.load(os.path.join(cache_key_dir, "styles.npy"))
    diams = np.load(os.path.join(cache_key_dir, "diams.npy")).item()

    # Load flows from separate files
    flows_dir = os.path.join(cache_key_dir, "flows")
    flows = [np.load(os.path.join(flows_dir, f)) for f in sorted(os.listdir(flows_dir))]

    return masks, flows, styles, diams
