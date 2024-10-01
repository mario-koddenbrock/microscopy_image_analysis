import os.path

import cv2
import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio import RasterioIOError


def pil_loader(image_path):
    """
    Load an image from a web URL or local path and convert it to RGB format.

    Args:
        image_path (str): Path to the image (can be a URL or local path).

    Returns:
        Image: The loaded and converted image.
    """
    if image_path.startswith('http://') or image_path.startswith('https://'):
        image = Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        image = Image.open(image_path)
    else:
        raise ValueError(f"Invalid image path: {image_path}")

    # Ensure the image is in RGB format (fixes issues with incorrect color channels)
    return image.convert("RGB")


def opencv_loader(path):
    # Read the image using OpenCV
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load image {path}")
    # Convert image to RGB if it's grayscale or BGR
    if len(image.shape) == 2:  # Grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:  # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    image = Image.fromarray(image)
    return image




def rasterio_loader(path):
    try:
        with rasterio.open(path) as src:
            image_array = src.read()  # Returns a NumPy array with shape (bands, rows, cols)
            # If the image has multiple bands, stack them appropriately
            if image_array.shape[0] == 1:
                # Single band (grayscale), stack to create RGB
                image_array = np.concatenate([image_array]*3, axis=0)
            elif image_array.shape[0] > 3:
                # More than 3 bands, take the first 3
                image_array = image_array[:3]
            # Transpose the array to (rows, cols, bands)
            image_array = np.transpose(image_array, (1, 2, 0))
            # Convert to uint8 if necessary
            if image_array.dtype != np.uint8:
                image_array = (image_array / np.max(image_array) * 255).astype(np.uint8)
            # Convert to PIL Image
            image = Image.fromarray(image_array)
            return image
    except RasterioIOError as e:
        raise ValueError(f"Failed to load image {path}: {e}")