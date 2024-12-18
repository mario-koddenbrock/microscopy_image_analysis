import glob
import os

import napari
import numpy as np
from cellpose import models, io
from segmentation_models_pytorch.metrics import precision

from mia.cellpose import evaluate_model, read_yaml
from mia.file_io import get_cellpose_ground_truth
from mia.utils import check_paths
from mia.viz import extract_cellpose_video


def view(
        image_path,
        param_file,
        output_dir,
        cache_dir="cache",
        show_gt=True,
        show_prediction=False,
        video_3d=True,
        show_viewer = True,
        export_video = False,
):

    params = read_yaml(param_file)
    check_paths(image_dir, output_dir, cache_dir)

    print(f"Processing image: {image_path}")
    results = evaluate_model(image_path, params, cache_dir)

    if results is None:
        print(f"Failed to process image {image_path}")

    image = results['image']
    masks = results['masks']
    ground_truth = results['ground_truth']
    are = results['are']
    precision = results['precision']
    recall = results['recall']
    f1 = results['f1']

    print(f"shape: {image.shape}")
    print(f"dtype: {image.dtype}")
    print(f"range: ({np.min(image)}, {np.max(image)})")
    print(f"are: {are}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1: {f1}")

    # plot the intensity distribution of the image
    # plot_intensity(image)

    # Initialize the Napari viewer
    viewer = napari.Viewer()

    # get the interquartile range of the intensities
    q1, q3 = np.percentile(image, [5, 99])

    # Add the image to the viewer
    viewer.add_image(
        image,
        contrast_limits=[q1, q3],
        name='Organoids',
        colormap='gray',
    )

    if show_gt and (ground_truth is not None):
        layer = viewer.add_labels(
            ground_truth,
            name="Ground truth",
            opacity=0.7,
            blending='translucent',
            # colormap='magma',
        )
        layer.contour = 2

    if show_prediction:
        # Add the labels to the viewer
        layer = viewer.add_labels(
            masks,
            name=params.model_name,
            opacity=0.7,
            blending='translucent',
            # colormap='magma',
        )
        layer.contour = 2

    if export_video:
        # Save the animation
        image_name = os.path.basename(image_path).replace(".tif", "")
        video_filename = f"{image_name}_{type}.mp4"

        if show_prediction:
            model_name = params.model_name
            video_filename = video_filename.replace(".mp4", f"_{model_name}.mp4")

        if show_gt:
            video_filename = video_filename.replace(".mp4", "_GT.mp4")

        if video_3d:
            video_filename = video_filename.replace(".mp4", "_3D.mp4")

        mode = "3D" if video_3d else "2D"
        num_z_slices = image.shape[0]
        extract_cellpose_video(viewer, output_dir, video_filename, num_z_slices, mode=mode)

        # Close the viewer to release resources
        viewer.close()

    if show_viewer:
        # setting the viewer to the center of the image
        center = image.shape[0] // 2
        viewer.dims.set_point(0, center)

        napari.run()





if __name__ == "__main__":

    main_folder = "Datasets/P013T/"

    # get all the subfolder
    subfolders = glob.glob(os.path.join(main_folder, "*"))

    for folder in subfolders:

        if not os.path.isdir(folder):
            continue

        # directory containing the images
        image_dir = os.path.join(folder, "images_cropped_isotropic")

        # directory to save the output
        output_dir = image_dir.replace("Datasets", "Segmentation")

        # Get the list of images
        image_paths = glob.glob(os.path.join(image_dir, "*.tif"))

        # Loop over the images
        for image_idx, image_path in enumerate(image_paths):

            image_name = os.path.basename(image_path).replace(".tif", "")
            param_file = os.path.join(folder, f"{image_name}.yaml")
            view(image_dir, output_dir, video_3d=False, show_gt=True, show_prediction=True, show_viewer=True, export_video=False)
