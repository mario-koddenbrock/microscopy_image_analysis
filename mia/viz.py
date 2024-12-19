import os

import cv2
import napari
import numpy as np
import pandas as pd
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

def show_napari(results, params):

    # Initialize the Napari viewer
    viewer = napari.Viewer()

    # Add the image to the viewer
    viewer.add_image(
        results['image'],
        contrast_limits=[113, 1300],
        name='Organoids',
        colormap='gray',
    )
    # Add the labels to the viewer
    viewer.add_labels(
        results['masks'],
        name=params.model_name,
        opacity=0.8,
        blending='translucent',
    )
    if results['ground_truth'] is not None:
        viewer.add_labels(
            results['ground_truth'],
            name='Ground Truth',
            opacity=0.8,
            blending='translucent',
        )
    # setting the viewer to the center of the image
    center = results['image'].shape[0] // 2
    viewer.dims.set_point(0, center)
    napari.run()


def plot_intensity(image):
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



def plot_aggregated_metric_variation(file_path, metric='f1', boxplot=False):
    """
    Detect varying parameters and plot the aggregated metric over these parameters with uncertainty bands
    aggregated over all image_name and type combinations. Optionally display a boxplot.

    Parameters:
        file_path (str): Path to the CSV file containing experiment results.
        metric (str): The column name of the metric to evaluate (default is 'f1').
        boxplot (bool): If True, display boxplots instead of error bars (default is False).
    """
    # Load data
    df = pd.read_csv(file_path)

    # Create output directory in the same folder as the input file
    output_dir = os.path.dirname(file_path)

    # Identify varying parameters (excluding fixed columns and specified metrics)
    excluded_columns = ['image_name', 'type', metric, 'duration', 'are', 'precision', 'recall', 'f1', 'jaccard_sklearn', 'jaccard_cellpose', 'jaccard']
    varying_columns = [col for col in df.columns if df[col].nunique() > 1 and col not in excluded_columns]

    if not varying_columns:
        print("No varying parameters detected.")
        return

    print("Varying parameters detected:", varying_columns)

    # Aggregate metric over all image_name and type
    for param in varying_columns:
        fig, ax = plt.subplots(figsize=(6, 4))

        if boxplot:
            # Create boxplot for the metric grouped by the parameter
            df.boxplot(column=metric, by=param, grid=False, ax=ax)
            # ax.set_title(f"Boxplot of {metric} vs {param} (over all images)")

            if len(pd.unique(df[param])) > 10:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')

            plt.xticks(rotation=45, ha='right')
            ax.set_xlabel(param)
            ax.set_ylabel(metric)
            plt.title("")
            plt.suptitle("")  # Remove default title
            output_path = os.path.join(output_dir, f"boxplot_{param}_{metric}.png")

        else:
            # Plot mean and standard deviation as error bars
            grouped = df.groupby(param)[metric].agg(['mean', 'std']).reset_index()
            x = grouped[param]
            y = grouped['mean']
            yerr = grouped['std']

            ax.errorbar(x, y, yerr=yerr, fmt='-o', capsize=5, label=f"{metric} (mean ± std)")


            if len(x) > 10:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            ax.set_xlabel(param)
            ax.set_ylabel(f"Aggregated {metric}")
            # ax.set_title(f"Aggregated {metric} vs {param} (over all images)")
            ax.legend()
            ax.grid(True)
            output_path = os.path.join(output_dir, f"errorbar_{param}_{metric}.png")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()
        plt.close(fig)
        print(f"Saved plot to {output_path}")



def plot_best_scores_barplot(file_path, metric='f1', output_file='best_scores_barplot.png'):
    """
    Visualize the best score for each image_name and type as a grouped bar plot.

    Parameters:
        file_path (str): Path to the CSV file containing experiment results.
        metric (str): The column name of the metric to visualize (default is 'f1').
        output_file (str): Path to save the bar plot (default is 'best_scores_barplot.png').
    """
    # Load data
    df = pd.read_csv(file_path)

    # Ensure the metric column exists
    if metric not in df.columns:
        print(f"Error: '{metric}' is not a valid column in the file.")
        print("Available columns:", ', '.join(df.columns))
        return

    # Find the best configuration per image_name and type based on the metric
    best_configs = df.loc[df.groupby(['image_name', 'type'])[metric].idxmax()]

    # Prepare data for plotting
    grouped = best_configs.groupby(['image_name', 'type'])[metric].max().unstack(fill_value=0)
    grouped.plot(kind='bar', figsize=(12, 6), alpha=0.8, edgecolor='black')

    # Plot customization
    # plt.title(f"Best {metric} Scores for Each Image and Type")
    plt.xlabel("Image Name")
    plt.ylabel(f"Best {metric} Score")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file)
    plt.show()
    print(f"Saved bar plot to {output_file}")
