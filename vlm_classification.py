import torch

from mia.plotting import plot_image_grid
from mia.utils import initialize_evaluators, evaluate_dataset
from vlm.clip import CLIPEvaluator

if __name__ == "__main__":
    # Determine the device to use (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Configuration for evaluators
    EVALUATORS_CONFIG = {
        "CLIP": CLIPEvaluator,
        # "PaliGemma": PaliGemmaEvaluator,
        # "PhiVision": PhiVisionEvaluator,
        # "QwenVL": QwenVLEvaluator,
        # "Vilt": ViltEvaluator,
        # "VisualBert": VisualBertEvaluator,
        # "BLIP": BLIPEvaluator,
        # "BLIP2": BLIP2Evaluator,
    }

    # Initialize evaluators
    evaluators = initialize_evaluators(device, EVALUATORS_CONFIG)

    dataset_name = "Digital Image of Bacterial Species"
    dataset_description = """
    The dataset from the study "Deep learning approach to bacterial colony classification" comprises 660 images representing 33 different genera and species of bacteria. 
    This dataset, called DIBaS (Digital Image of Bacterial Species), was created for bacterial classification using deep learning methods. 
    The images were taken with a microscope and analyzed using Convolutional Neural Networks (CNNs) and machine learning classifiers like Support Vector Machines (SVM) and Random Forest. 
    The dataset is publicly available for research purposes, allowing for advancements in bacterial recognition systems.
    """
    dataset_path = "datasets/Classification"
    num_images_per_class = 1
    num_classes = 33

    results = evaluate_dataset(
        evaluators,
        dataset_name,
        dataset_description,
        dataset_path,
        num_images_per_class,
        num_classes,
    )

    plot_image_grid(results, num_classes, image_name="vlm_classification_results.png")
