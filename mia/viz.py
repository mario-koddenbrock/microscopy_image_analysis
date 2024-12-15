import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from napari_animation import Animation
from tqdm import tqdm


# import matplotlib
# matplotlib.use('TkAgg')


def extract_cellpose_video(viewer, output_dir, video_filename, num_z_slices, mode='2D', rotation_steps=72):
    """
    Create a video from a Napari viewer, with optional 2D z-slice animation or 3D rotation.

    Parameters:
        viewer: napari.Viewer
            The Napari viewer instance.
        output_dir: str
            Directory to save the video.
        video_filename: str
            Name of the output video file.
        num_z_slices: int
            Number of z-slices for the 2D animation.
        mode: str
            Animation mode, either '2D' or '3D'. Defaults to '2D'.
        rotation_angle: int
            Total angle (in degrees) to rotate in 3D mode. Defaults to 360 degrees.
        rotation_steps: int
            Number of steps (frames) for the 3D rotation animation. Defaults to 36.
    """
    video_path = os.path.join(output_dir, video_filename)

    # Set FPS based on mode
    fps = 100 if mode == '2D' else 30  # Higher FPS for 2D, lower for 3D

    # Create an animation object
    animation = Animation(viewer)

    if mode == '2D':
        # 2D animation: iterate through z-slices
        for z in tqdm(range(int(num_z_slices / 2)), desc="Capturing 2D frames"):
            viewer.dims.set_point(0, 2 * z)  # Set the z-slice
            animation.capture_keyframe()

    elif mode == '3D':
        # Ensure viewer is in 3D mode
        viewer.dims.ndisplay = 3

        # 3D animation: rotate the camera
        for i in tqdm(range(rotation_steps), desc="Capturing 3D frames"):
            azimuth = (i * 360) / rotation_steps
            viewer.camera.angles = (0, azimuth, 0)  # Adjust elevation and azimuth
            animation.capture_keyframe()

    else:
        raise ValueError("Invalid mode. Use '2D' or '3D'.")

    # Save the animation
    animation.animate(video_path, canvas_only=True, fps=fps, quality=9)
    print(f"Saved animation to {video_path}")


def plot_intensety(image):
    plt.hist(image.flatten(), bins=1000)
    # make log scale
    plt.yscale('log')
    plt.ylabel('count')
    plt.xlabel('intensity')
    plt.title('intensity distribution')
    plt.show()

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
