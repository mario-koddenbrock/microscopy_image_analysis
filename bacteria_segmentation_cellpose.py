import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from cellpose.models import Cellpose
from csbdeep.utils import normalize
from skimage.io import imread
from skimage.measure import regionprops
from stardist import random_label_cmap, _draw_polygons
from stardist.plot import render_label

from mia.viz import save_as_video


def main(image_dir, output_dir):
    """Main function to load images, segment them and visualize results."""
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # DEFINE CELLPOSE MODEL
    # model_type='cyto' or model_type='nuclei'
    model = Cellpose(gpu=False, model_type="bact_omni")
    flow_threshold = 0.3
    diameter = None
    cellprob_threshold = 1 # -6, .. 6
    channels = [0, 0]  # grayscale image

    image_paths = glob.glob(os.path.join(image_dir, "*.png"))

    for image_idx, image_path in enumerate(image_paths):

        image_name = os.path.basename(image_path).replace(".png", "")
        print(f"Processing image: {image_name}")

        image = imread(image_path, as_gray=True)

        # normalize image to 0..255
        img = normalize(image, 1, 99.8)

        # ref_img = normalize(ref_img)

        masks, flows, styles, diams = model.eval(
            img,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            channels=channels,
        )


        # Get region properties of the labels for finding centroid of each instance
        regions = regionprops(masks)

        image_with_labels = render_label(masks, img=image, alpha=0.7, normalize_img=False, alpha_boundary=1)
        # image_with_labels = labels
        # im = Image.fromarray(image_with_labels[:, :, :3])
        # im.save(os.path.join(output_dir, os.path.basename(image_path)))

        # Alternatively, save using PIL (requires conversion to uint8 format)
        image_with_labels_path = os.path.join(output_dir, f"{image_name}_segmentation_overlay.png")
        labels_8bit = (masks / masks.max() * 255).astype(np.uint8)
        Image.fromarray(labels_8bit).save(image_with_labels_path)

        image_with_polys_path = os.path.join(output_dir, f"{image_name}_poly_overlay.png")
        plt.figure(figsize=(100, 100))
        img_show = img if img.ndim == 2 else img[..., 0]
        # coord, points, prob = details['coord'], details['points'], details['prob']
        plt.imshow(img_show, cmap='gray')
        plt.axis('off')
        a = plt.axis()
        # _draw_polygons(coord, points, prob, show_dist=True)
        plt.axis(a)
        plt.tight_layout()
        plt.savefig(image_with_polys_path)
        plt.close()

        if image_idx > -1:
            video_path = os.path.join(output_dir, f"{image_name}_segmentation_count_video.mp4")
            save_as_video(video_path, image_with_labels, masks, regions)




if __name__ == "__main__":
    image_dir = "Datasets/Run_1/S.aureus"
    output_dir = "Segmentation/Run_1/S.aureus"
    main(image_dir, output_dir)

    image_dir = "Datasets/Run_1/E.coli"
    output_dir = "Segmentation/Run_1/E.coli"
    main(image_dir, output_dir)
