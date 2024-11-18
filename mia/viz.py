import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


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
