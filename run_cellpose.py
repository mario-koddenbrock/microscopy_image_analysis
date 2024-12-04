import glob
import os
import cellpose
import matplotlib
import napari
import numpy as np
from cellpose import models, io
from napari_animation import Animation
from skimage.exposure import rescale_intensity
from mia.hash import compute_hash, load_from_cache, save_to_cache

# Set the path to the ffmpeg executable - only needed for exporting animations
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
print(f"Cellpose version: {cellpose.version}")

# Do you want to show the viewer or export the video?
# TODO: they are mutually exclusive right now
show_viewer = True
export_video = False

def view(image_dir, output_dir, cache_dir="cache"):

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # List of channels, either of length 2 or of length number of images by 2.
    # First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
    # Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
    # For instance, to segment grayscale images, input [0,0].
    # To segment images with cells in green and nuclei in blue, input [2,3].
    # To segment one grayscale image and one image with cells in green and nuclei in blue, input [[0,0], [2,3]].
    # Defaults to [0,0].
    channels = [0, 0]

    # Load the Cellpose model
    model_dict = {
        "cyto": models.Cellpose(model_type='cyto', gpu=False),
        "cyto2": models.Cellpose(model_type='cyto2', gpu=False),
        "cyto3": models.Cellpose(model_type='cyto3', gpu=False),
        "nuclei": models.Cellpose(model_type='nuclei', gpu=False),
    }

    # Get the list of images
    image_paths = glob.glob(os.path.join(image_dir, "*.tif"))

    # Loop over the images
    for image_idx, image_path in enumerate(image_paths[1:]):

        image_name = os.path.basename(image_path).replace(".tif", "")

        # Read image with cellpose.io
        image = io.imread(image_path)

        # TODO: contrast normalization on image
        image = rescale_intensity(image)

        print(f"image: {image_name}")
        print(f"shape: {image.shape}")
        print(f"dtype: {image.dtype}")
        print(f"range: ({np.min(image)}, {np.max(image)})")

        # iterate over all cellpose models
        for model_name, model in model_dict.items():
            print(f"Running model: {model_name}")

            parameters = {
                "model_name": model_name,
                "diameter": None,
                "channels": channels,
                "do_3D": True,
                "z_axis": 0,
            }

            cache_key = compute_hash(image, parameters)
            cached_result = load_from_cache(cache_dir, cache_key)

            if cached_result:
                print(f"Loaded cached result for {model_name}.")
                masks, flows, styles, diams = cached_result
            else:
                print(f"No cache found. Running eval for {model_name}.")
                masks, flows, styles, diams = model.eval(
                    image,
                    diameter=None,
                    channels=channels,
                    do_3D=True,
                    z_axis=0
                )
                save_to_cache(cache_dir, cache_key, masks, flows, styles, diams)

            # Initialize the Napari viewer
            viewer = napari.Viewer()

            # Add the image to the viewer
            viewer.add_image(
                image,
                contrast_limits=[113, 1300],
                name='Organoids',
                colormap='gray',
            )


            # Add the labels to the viewer
            viewer.add_labels(
                masks,
                name=model_name,
                opacity=0.8,
                blending='translucent',
                # colormap=matplotlib.cm.get_cmap('Set1'),
            )

            if export_video:
                # Create an animation object
                animation = Animation(viewer)

                # Loop over z-slices and capture frames
                for z in range(image.shape[0]):
                    viewer.dims.set_point(0, z)
                    animation.capture_keyframe()

                # Save the animation
                video_filename = f"{image_name}_{model_name}_animation.mp4"
                video_path = os.path.join(output_dir, video_filename)
                animation.animate(video_path, canvas_only=False)
                print(f"Saved animation to {video_path}")

                # Close the viewer to release resources
                viewer.close()

            if show_viewer:
                # setting the viewer to the center of the image
                center = image.shape[0] // 2
                viewer.dims.set_point(0, center)

                napari.run()





        if image_idx > 10:
            break


if __name__ == "__main__":

    # directory containing the images
    image_dir = "Datasets/Organoids/20230712_P013T_cropped_isotropic/images"

    # directory to save the output
    output_dir = "Segmentation/Organoids/20230712_P013T_cropped_isotropic"

    view(image_dir, output_dir)
