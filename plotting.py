import os.path
from matplotlib import pyplot as plt

from file_io import pil_loader


def show_prediction_result(image_path, dataset_name, ground_truth, results):
    """
    Show image with ground truth label and the predicted label of selected VLMs.

    Args:
        image_path (str): Path to the image (can be a URL or local path).
        dataset_name (str): Name of the dataset.
        ground_truth (str): Ground truth label for the image.
        results (dict): Dictionary of VLM results with VLM names as keys and their outputs as values.
    """

    image = pil_loader(image_path)

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
        fontsize=8,  # Smaller font size
        verticalalignment='bottom',  # Align text box at the bottom
        horizontalalignment='right',  # Align text box to the right
        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5}  # White background with transparency
    )

    # Save the image with the predictions
    image_name = f"{dataset_name}_{ground_truth}.png"
    save_to = os.path.join("results", image_name)

    # Save the plot to the specified output file
    plt.tight_layout()
    plt.savefig(save_to, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Results plotted and saved to {save_to}")