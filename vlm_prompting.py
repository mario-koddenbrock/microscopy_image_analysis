import warnings

import torch

from utils import initialize_evaluators, evaluate_image
from vlm.blip_2 import BLIP2Evaluator
from vlm.blip import BLIPEvaluator
from vlm.clip import CLIPEvaluator
from vlm.pali_gemma import PaliGemmaEvaluator
from vlm.phi_vision import PhiVisionEvaluator
from vlm.qwen_vl import QwenVLEvaluator
from vlm.vilt import ViltEvaluator
from vlm.visual_bert import VisualBertEvaluator

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

    # Iterate over each dataset and evaluate
    for image_path in image_pathes:
        evaluate_image(evaluators, image_path, prompt)