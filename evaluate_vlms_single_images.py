import warnings

import torch

from datasets import DATASET_DESCRIPTIONS, evaluate_dataset, evaluate_image
from utils import initialize_evaluators
from vlm.blip_2 import BLIP2Evaluator
from vlm.blip import BLIPEvaluator
from vlm.clip import CLIPEvaluator
from vlm.pali_gemma import PaliGemmaEvaluator
from vlm.phi_vision import PhiVisionEvaluator
from vlm.qwen_vl import QwenVLEvaluator
from vlm.vilt import ViltEvaluator
from vlm.visual_bert import VisualBertEvaluator

import huggingface as hf

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

    # Log in to Hugging Face
    hf.huggingface_login()

    # Configuration for evaluators
    EVALUATORS_CONFIG = {
        "CLIP": CLIPEvaluator,
        "PaliGemma": PaliGemmaEvaluator,
        "PhiVision": PhiVisionEvaluator,
        "QwenVL": QwenVLEvaluator,
        "Vilt": ViltEvaluator,
        "VisualBert": VisualBertEvaluator,
        "BLIP": BLIPEvaluator,
        "BLIP2": BLIP2Evaluator,
    }

    # # Check model sizes and cache status
    # for model_name, evaluator_class in EVALUATORS_CONFIG.items():
    #     model_id = evaluator_class().model_id
    #     size_in_bytes = hf.get_model_size(model_id)
    #     is_in_cache = hf.check_model_in_cache(model_id)
    #     print(f"Model: {model_name}")
    #     print(f"Size: {hf.format_size(size_in_bytes)}")
    #     print(f"In Cache: {'Yes' if is_in_cache else 'No'}")

    # Initialize evaluators
    evaluators = initialize_evaluators(device, EVALUATORS_CONFIG)

    # Iterate over each dataset and evaluate
    for image_path in image_pathes:
        evaluate_image(evaluators, image_path)