import warnings

import torch

from mia.plotting import plot_image_grid
from mia.utils import initialize_evaluators, evaluate_image
from vlm.pali_gemma import PaliGemmaEvaluator

# Ignore warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    image_pathes = [
        "https://www.uni-assist.de/fileadmin/_processed_/d/5/csm_htw-berlin_Friederike_Coenen_36e250c32e.jpg",
        "https://www.htw-berlin.de/files/Presse/_tmp_/2/2/csm_HTW-Berlin-HTW-Imagefotos-DSC216129-HTW_Berlin-Alexander_Rentsch_f044c85c99.jpg",
        ]

    # Determine the device to use (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")


    # Configuration for evaluators
    EVALUATORS_CONFIG = {
        # "CLIP": CLIPEvaluator,
        "PaliGemma": PaliGemmaEvaluator,
        # "PhiVision": PhiVisionEvaluator,
        # "QwenVL": QwenVLEvaluator,
        # "Vilt": ViltEvaluator,
        # "VisualBert": VisualBertEvaluator,
        # "BLIP": BLIPEvaluator,
        # "BLIP2": BLIP2Evaluator,
    }

    # Initialize evaluators
    evaluators = initialize_evaluators(device, EVALUATORS_CONFIG)

    # Prompt to use for evaluation
    prompt = "Describe the image:"
    result_images = []

    # Iterate over each dataset and evaluate
    for image_path in image_pathes:
        result_image_path = evaluate_image(evaluators, image_path, prompt)
        result_images.append(result_image_path)

    plot_image_grid(result_images, 2, rows=1, image_name="vlm_prompting_results.png")