import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from csbdeep.utils import normalize
from skimage.io import imread
from skimage.measure import regionprops
from stardist import random_label_cmap, _draw_polygons
from stardist.data import test_image_nuclei_2d
from stardist.models import StarDist2D
from stardist.plot import render_label


def main(image_dir, output_dir):
    """Main function to load images, segment them and visualize results."""
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lbl_cmap = random_label_cmap()

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # prints a list of available models
    StarDist2D.from_pretrained()

    # creates a pretrained model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    image_paths = glob.glob(os.path.join(image_dir, "*.png"))

    for image_idx, image_path in enumerate(image_paths):

        image_name = os.path.basename(image_path).replace(".png", "")
        print(f"Processing image: {image_name}")

        image = imread(image_path, as_gray=True)
        # ref_img = test_image_nuclei_2d()

        # normalize image to 0..255
        img = normalize(image, 1, 99.8)

        # ref_img = normalize(ref_img)

        labels, details = model.predict_instances(img)

        # Get region properties of the labels for finding centroid of each instance
        regions = regionprops(labels)

        image_with_labels = render_label(labels, img=image, alpha=0.7, normalize_img=False, alpha_boundary=1)
        # image_with_labels = labels
        # im = Image.fromarray(image_with_labels[:, :, :3])
        # im.save(os.path.join(output_dir, os.path.basename(image_path)))

        # Alternatively, save using PIL (requires conversion to uint8 format)
        image_with_labels_path = os.path.join(output_dir, f"{image_name}_segmentation_overlay.png")
        labels_8bit = (labels / labels.max() * 255).astype(np.uint8)
        Image.fromarray(labels_8bit).save(image_with_labels_path)

        image_with_polys_path = os.path.join(output_dir, f"{image_name}_poly_overlay.png")
        plt.figure(figsize=(100, 100))
        img_show = img if img.ndim == 2 else img[..., 0]
        coord, points, prob = details['coord'], details['points'], details['prob']
        plt.imshow(img_show, cmap='gray')
        plt.axis('off')
        a = plt.axis()
        _draw_polygons(coord, points, prob, show_dist=True)
        plt.axis(a)
        plt.tight_layout()
        plt.savefig(image_with_polys_path)
        plt.close()

        if image_idx > -1:
            video_path = os.path.join(output_dir, f"{image_name}_segmentation_count_video.mp4")
            save_as_video(video_path, image_with_labels, labels, regions)


def save_as_video(output_video_path, image_with_labels, labels, regions):
    # Get the number of unique labels (excluding the background)
    unique_labels = np.unique(labels)
    num_instances = len(unique_labels) - 1  # Assuming 0 is the background
    # Directory to save frames temporarily
    os.makedirs('frames', exist_ok=True)
    # Create frames progressively increasing the count
    frame_files = []

    # sort the region list by y value
    y_values = [r.centroid[0] for r in regions]
    sorted_idx = np.argsort(y_values)
    regions = [regions[i] for i in sorted_idx]


    for count, region in enumerate(regions, start=1):
        # Create a figure for each count
        plt.figure(figsize=(10, 10))
        plt.imshow(image_with_labels)
        plt.axis('off')

        # Highlight the current instance with a large circle
        y, x = region.centroid
        plt.plot(x, y, marker='o', markersize=30, markeredgewidth=4, markeredgecolor='yellow', fillstyle='none')

        # Calculate length and width from the bounding box
        min_row, min_col, max_row, max_col = region.bbox
        length = max_row - min_row
        width = max_col - min_col

        # Display additional info about the region, including length and width
        info_text = f'Count: {count}\nArea: {region.area}\nEccentricity: {region.eccentricity:.2f}\nLength: {length}\nWidth: {width}'
        plt.text(10, 250, info_text, color='blue', fontsize=15, bbox=dict(facecolor='white', alpha=0.7))

        # Save the frame to disk
        frame_filename = f'frames/frame_{count:03d}.png'
        plt.savefig(frame_filename, bbox_inches='tight', pad_inches=0)
        frame_files.append(frame_filename)
        plt.close()

    # Create a video from the saved frames
    frame_rate = 1  # 1 frame per second
    # Load the first frame to get dimensions
    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    # Write each frame into the video
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        out.write(frame)
    # Release the VideoWriter and clean up
    out.release()
    # Optionally, delete the temporary frames
    for frame_file in frame_files:
        os.remove(frame_file)
    print(f"Video saved at {output_video_path}")


if __name__ == "__main__":
    image_dir = "Datasets/Run_1/S.aureus"
    output_dir = "Segmentation/Run_1/S.aureus"
    main(image_dir, output_dir)

    image_dir = "Datasets/Run_1/E.coli"
    output_dir = "Segmentation/Run_1/E.coli"
    main(image_dir, output_dir)
