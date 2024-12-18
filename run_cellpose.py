import glob
import os

import napari
import numpy as np
from cellpose import models, io

from mia.cellpose import evaluate_model
from mia.file_io import get_cellpose_ground_truth
from mia.utils import check_paths
from mia.viz import extract_cellpose_video


def view(image_dir, output_dir, cache_dir="cache", show_gt=True, show_prediction=False, video_3d = True):

    check_paths(image_dir, output_dir, cache_dir)

    # Do you want to show the viewer or export the video?
    # TODO: they are mutually exclusive right now
    show_viewer = True
    export_video = False

    type = "Membranes" # "Nuclei" or "Membranes"
    if type == "Nuclei":
        channel_idx = 0
    elif type == "Membranes":
        channel_idx = 1

    # Load the Cellpose model
    model_dict = {
        # "cyto": models.Cellpose(model_type='cyto', gpu=False),
        # "cyto2": models.Cellpose(model_type='cyto2', gpu=False),
        "cyto3": models.Cellpose(model_type='cyto3', gpu=False),
        "nuclei": models.Cellpose(model_type='nuclei', gpu=False),
    }

    # Get the list of images
    image_paths = glob.glob(os.path.join(image_dir, "*.tif"))

    # Loop over the images
    for image_idx, image_path in enumerate(image_paths):

        image_name = os.path.basename(image_path).replace(".tif", "")

        # Read image with cellpose.io
        image = io.imread(image_path)

        ground_truth = get_cellpose_ground_truth(image_path, image_name, type)

        # # TODO: contrast normalization on image
        # image = rescale_intensity(image)

        if image.ndim == 4:
            image = image[:, channel_idx, :, :]

        print(f"image: {image_name}")
        print(f"shape: {image.shape}")
        print(f"dtype: {image.dtype}")
        print(f"range: ({np.min(image)}, {np.max(image)})")

        # plot the intensity distribution of the image
        # plot_intensity(image)

        # iterate over all cellpose models
        for model_name, model in model_dict.items():
            print(f"Running model: {model_name}")

            params = {
                'model_name': model_name,
                'channel_segment': 0,
                'channel_nuclei': 0,
                'channel_axis': None,
                'invert': False,
                'normalize': True,
                'diameter': None,
                'do_3D': True,
            }

            print(f"Processing image {image_name} ({image.shape}) with parameters: {params}")
            if show_prediction:
                masks, flows, styles, diams = evaluate_model(model, image, params, cache_dir)

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
            #     simple_jaccard = simple_iou(ground_truth, masks)
            #     jaccard = jaccard_score(ground_truth, masks)
            #     fscore = f1_score(ground_truth, masks)
            #     # precision, recall, fscore = boundary_scores(ground_truth, masks)
            #
            #     print(f"Simple Jaccard: {simple_jaccard}")
            #     print(f"Jaccard: {jaccard}")
            #     print(f"F1-Score: {fscore}")
            #     # print(f"Boundaries: {boundaries}")

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

        # if image_idx > 10:
        #     break




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

        for video_3d in [True, False]:
            for show_gt in [True, False]:
                for show_prediction in [True, False]:
                    print(f"video_3d: {video_3d}, show_gt: {show_gt}, show_prediction: {show_prediction}")
                    try:
                        view(image_dir, output_dir, video_3d=video_3d, show_gt=show_gt, show_prediction=show_prediction)
                    except Exception as e:
                        print(f"Error: {e}")
                        continue
