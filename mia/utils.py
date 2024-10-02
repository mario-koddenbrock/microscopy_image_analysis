import os.path
import time

from mia import plotting, prompts
from vlm.clip import CLIPEvaluator
from vlm.visual_bert import VisualBertEvaluator


def initialize_evaluators(device, config):
    """
    Initialize evaluators based on the provided configuration.

    Args:
        device (str): The device to run the evaluators on (e.g., 'cpu' or 'cuda').
        config (dict): Configuration dictionary mapping evaluator names to their classes.

    Returns:
        dict: A dictionary of initialized evaluators.
    """
    evaluators = {}
    for name, evaluator_class in config.items():
        start_time = time.time()
        evaluators[name] = evaluator_class(device=device)
        end_time = time.time()
        print(f"Initialized {name} in {end_time - start_time:.2f} seconds")
    return evaluators


def evaluate_image(evaluators, image_path, sample_prompt = "Describe the image:"):
    """
    Evaluate an image using the provided evaluators.

    Args:
        sample_prompt (str): The prompt to use for evaluation.
        evaluators (dict): Dictionary of initialized evaluators.
        image_path (str): Path to the image to evaluate.
    """

    results = {name: "default_result" for name in evaluators.keys()}
    for name, evaluator in evaluators.items():
        start_time = time.time()
        results[name] = evaluator.evaluate(prompt=sample_prompt, image_path=image_path)
        end_time = time.time()
        print(f"{name} Output: {results[name]} (Evaluated in {end_time - start_time:.2f} seconds)")

    image_name = os.path.basename(image_path)
    return plotting.show_prediction_result(image_path, image_name, sample_prompt, results)



def evaluate_dataset(evaluators, dataset_name, dataset_description, dataset_path, num_images_per_class, num_classes):
    """
    Evaluate a dataset using the provided evaluators.

    Args:
        evaluators (dict): Dictionary of initialized evaluators.
        dataset_name (str): Name of the dataset to evaluate.
        dataset_description (str): Description of the dataset.
        dataset_path (str): Path to the dataset to evaluate.
        num_images (int): How many images should be evaluated.
    """
    classes = get_dataset_classes(dataset_path)
    if not classes:
        print(f"No classes found for dataset: {dataset_path}")
        return

    print(f"Dataset: {dataset_name}")
    prompt = prompts.classification_prompt(dataset_name=dataset_name, dataset_description=dataset_description)

    result_images = []
    for class_idx, class_name in enumerate(classes[:num_classes]):
        print(f"\tClass {class_idx}: {class_name}")
        results = {name: "default_result" for name in evaluators.keys()}
        class_files = get_class_files(dataset_path, class_name)

        for sample_image_url in class_files[:num_images_per_class]:
            print(f"\tSample image: {sample_image_url}")

            for name, evaluator in evaluators.items():

                if isinstance(evaluator, CLIPEvaluator) or isinstance(evaluator, VisualBertEvaluator):
                    evaluator.set_class_names(classes)

                start_time = time.time()
                results[name] = evaluator.evaluate(prompt=prompt, image_path=sample_image_url)
                end_time = time.time()
                print(f"\t\t{name} Output: {results[name]} (Evaluated in {end_time - start_time:.2f} seconds)")

            result_image_path = plotting.show_prediction_result(sample_image_url, dataset_name, class_name, results)
            result_images.append(result_image_path)

    return result_images

def get_class_files(dataset_path, class_name):

    classes = os.listdir(dataset_path)

    # if folder contains test and train folders, only use test
    if "test" in classes:
        dataset_path = os.path.join(dataset_path, "test")

    # if folder contains images, only use images
    if "images" in classes:
        dataset_path = os.path.join(dataset_path, "images")

    # get all images in dataset_path/class_name
    class_path = os.path.join(dataset_path, class_name)

    if not os.path.exists(class_path):
        raise ValueError(f"Class {class_name} not found in dataset {dataset_path}")

    class_subfolders = os.listdir(class_path)

    # if folder contains test and train folders, only use test
    if "test" in class_subfolders:
        class_path = os.path.join(class_path, "test")

    # get full file path to the images inside class_path
    files = [os.path.join(class_path, f) for f in os.listdir(class_path) if not f.startswith(".")]

    return files


def get_dataset_classes(dataset_path):

    # get the folder names inside the dataset
    folder = os.listdir(dataset_path)

    # if folder contains test and train folders, only use test
    if "test" in folder:
        folder = os.listdir(os.path.join(dataset_path, "test"))

    # if folder contains images, only use images
    if "images" in folder:
        folder = os.listdir(os.path.join(dataset_path, "images"))

    # filter out the folders that are not classes or are not visible
    classes = [f for f in folder if not f.startswith(".")]

    return classes




