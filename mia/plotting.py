import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets

from mia import file_io


def show_prediction_result(image_path, dataset_name, ground_truth, results):
    """
    Show image with ground truth label and the predicted label of selected VLMs.

    Args:
        image_path (str): Path to the image (can be a URL or local path).
        dataset_name (str): Name of the dataset.
        ground_truth (str): Ground truth label for the image.
        results (dict): Dictionary of VLM results with VLM names as keys and their outputs as values.
    """

    image = file_io.pil_loader(image_path)

    # Set up the figure and axes for plotting
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(image)
    ax.axis('off')  # Hide axes

    # Prepare the text to display (ground truth and VLM results)
    text_str = f"Ground Truth: {ground_truth}\n\n"
    for vlm_name, vlm_result in results.items():
        text_str += f"{vlm_name}: {vlm_result}\n\n"

    # Add the text box in the bottom-right corner
    ax.text(
        0.95, 0.05, text_str,
        transform=ax.transAxes,  # Position relative to axes
        fontsize=15,  # Smaller font size
        verticalalignment='bottom',  # Align text box at the bottom
        horizontalalignment='right',  # Align text box to the right
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5},  # White background with transparency
        wrap=True,  # Wrap text if it exceeds the width of the image
    )

    # filename of the input image
    image_name = os.path.basename(image_path)
    image_name = image_name.replace(".png", "")
    image_name = image_name.replace(".tif", "")
    image_name = image_name.replace(".jpg", "")

    # Replace spaces and colons in the ground truth label
    ground_truth = ground_truth.replace(" ", "_")
    ground_truth = ground_truth.replace(":", "_")

    # Save the image with the predictions
    image_name = f"{ground_truth}_{image_name}.png"
    save_to = os.path.join("results", "Classification", image_name)

    # Save the plot to the specified output file
    plt.tight_layout()
    plt.savefig(save_to, bbox_inches='tight', dpi=300)
    plt.close()

    # print(f"\t\tResults plotted and saved to {save_to}")
    return save_to



def visualize_class_samples(dataset, class_names, num_classes=8, image_size=(224, 224), save_dir='results/Classification', save_filename='class_samples.png'):
    """
    Visualizes one sample image from each class in a grid (5 rows per column) and saves the plot.

    :param dataset: The dataset object (e.g., ImageFolder) from which to sample images.
    :param class_names: List of class names.
    :param num_classes: Number of classes to display in the grid.
    :param image_size: Size of each image in the grid.
    :param save_dir: Directory where the image will be saved.
    :param save_filename: Filename for the saved image.
    """
    # Dictionary to store one image per class
    class_images = {}

    # Loop over the dataset to collect one image per class
    for path, label in dataset.samples:  # dataset.samples contains paths and labels
        class_name = class_names[label]
        if class_name not in class_images:
            class_images[class_name] = file_io.rasterio_loader(path)  # Load image using rasterio_loader
        if len(class_images) == num_classes:
            break

    # Create a list of images in the correct order based on class_names
    image_list = [class_images[class_name] for class_name in class_names[:num_classes]]

    # Determine number of rows (5) and columns (calculated based on num_classes)
    num_rows = 5
    num_cols = (num_classes + num_rows - 1) // num_rows  # Ensures we get enough columns

    # Set up the grid for plotting the images
    fig, axes = plt.subplots(num_cols, num_rows, figsize=(num_cols * 2, num_rows * 3))

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for i, (img, ax) in enumerate(zip(image_list, axes)):
        # Display the image loaded from rasterio
        ax.imshow(img)
        ax.axis('off')

        # Set the title to the class name
        ax.set_title(class_names[i], fontsize=12)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        axes[j].set_xticklabels([])
        axes[j].set_yticklabels([])

    # Adjust layout
    plt.subplots_adjust(wspace=0, hspace=0.25)

    # Create the results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")

    # Show the plot
    plt.show()


def plot_image_grid(image_list, num_classes=33, figsize=(20, 20), rows=5, image_name="vlm_results.png"):
    """
    Plots a grid of images from a list.

    Parameters:
    image_list (list): A list of image arrays (as numpy arrays or PIL images).
    figsize (tuple): Size of the figure (optional).
    """

    cols = (num_classes + rows - 1) // rows  # Ensures we get enough columns

    fig, axes = plt.subplots(cols, rows, figsize=figsize)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(image_list):
            # Check if the item in the list is an image path or an already loaded image
            if isinstance(image_list[i], str):
                # If it's a string (i.e., a file path), load the image
                img = file_io.pil_loader(image_list[i])
            else:
                # Otherwise, assume it's already an image array
                img = image_list[i]

            # Convert the image to a NumPy array and plot it
            ax.imshow(np.array(img))
        ax.axis('off')  # Hide the axes


    save_to = os.path.join("results", "Classification", image_name)

    # Save the plot to the specified output file
    plt.tight_layout()
    plt.savefig(save_to, bbox_inches='tight', dpi=300)
    # plt.close()


if __name__ == '__main__':
    data_dir = os.path.join('../datasets', 'Classification')

    # Load the dataset (without splitting) to visualize images
    full_dataset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
    class_names = full_dataset.classes

    # Visualize one image per class in a grid and save it
    visualize_class_samples(full_dataset, class_names, num_classes=len(class_names))
