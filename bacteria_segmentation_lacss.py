import glob
import os

import cv2
import lacss
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from csbdeep.utils import normalize
from lacss.deploy.predict import Predictor
from skimage.io import imread
from skimage.measure import regionprops
from stardist import random_label_cmap, _draw_polygons
from stardist.models import StarDist2D
from stardist.plot import render_label
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb

from lacss.utils import show_images

def main(image_dir, output_dir):
    """Main function to load images, segment them and visualize results."""

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lbl_cmap = random_label_cmap()

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    predictor = Predictor(lacss.deploy.model_urls["lacss3-base"])
    image_paths = glob.glob(os.path.join(image_dir, "*.png"))

    for image_idx, image_path in enumerate(image_paths):

        image_name = os.path.basename(image_path).replace(".png", "")
        print(f"Processing image: {image_name}")

        image = imageio.imread(image_path)

        labels = predictor.predict(
            image, reshape_to=[230, 307], nms_iou=0.4,
        )["pred_label"]

        show_images([
            image,
            labels,
        ])

        # # Get region properties of the labels for finding centroid of each instance
        # regions = regionprops(labels)

        image_with_labels = render_label(labels, img=image, alpha=0.7, normalize_img=False, alpha_boundary=1)
        # image_with_labels = labels
        im = Image.fromarray(image_with_labels[:, :, :3])
        save_to = os.path.join(output_dir, f"{image_name}_segmentation_overlay.png")
        im.save(save_to)
        print(f"Saved image with labels: {save_to}")

        # Alternatively, save using PIL (requires conversion to uint8 format)
        image_with_labels_path = os.path.join(output_dir, f"{image_name}_segmentation_overlay.png")
        labels_8bit = (labels / labels.max() * 255).astype(np.uint8)
        Image.fromarray(labels_8bit).save(image_with_labels_path)

        # if image_idx > -1:
        #     video_path = os.path.join(output_dir, f"{image_name}_segmentation_count_video.mp4")
        #     save_as_video(video_path, image_with_labels, labels, regions)



if __name__ == "__main__":
    image_dir = "Datasets/Run_1/S.aureus"
    output_dir = "Segmentation/Run_1/S.aureus"
    main(image_dir, output_dir)

    image_dir = "Datasets/Run_1/E.coli"
    output_dir = "Segmentation/Run_1/E.coli"
    main(image_dir, output_dir)
