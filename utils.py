import os.path
import time

import cv2
import numpy as np
import rasterio
from PIL import Image
from rasterio.errors import RasterioIOError

import plotting
import prompts
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


def evaluate_image(evaluators, image_path):
    """
    Evaluate an image using the provided evaluators.

    Args:
        evaluators (dict): Dictionary of initialized evaluators.
        image_path (str): Path to the image to evaluate.
    """
    # sample_prompt = prompts.classification_prompt() # TODO: Implement this
    sample_prompt = "Describe the image:"

    results = {name: "default_result" for name in evaluators.keys()}
    for name, evaluator in evaluators.items():
        start_time = time.time()
        results[name] = evaluator.evaluate(prompt=sample_prompt, image_path=image_path)
        end_time = time.time()
        print(f"{name} Output: {results[name]} (Evaluated in {end_time - start_time:.2f} seconds)")

    image_name = os.path.basename(image_path)
    plotting.show_prediction_result(image_path, image_name, "Unknown", results)



def evaluate_dataset(evaluators, dataset_name, dataset_description):
    """
    Evaluate a dataset using the provided evaluators.

    Args:
        evaluators (dict): Dictionary of initialized evaluators.
        dataset_name (str): Name of the dataset to evaluate.
        dataset_description (str): Description of the dataset.
    """
    classes = get_dataset_classes(dataset_name)
    if not classes:
        print(f"No classes found for dataset {dataset_name}")
        return

    print(f"Dataset: {dataset_name}")
    sample_prompt = prompts.classification_prompt(dataset_name=dataset_name, dataset_description=dataset_description)

    for class_name in classes:
        print(f"Class: {class_name}")
        results = {name: "default_result" for name in evaluators.keys()}
        class_files = get_class_files(dataset_name, class_name)
        sample_image_url = class_files[0]

        print(f"Sample image: {sample_image_url}")

        for name, evaluator in evaluators.items():

            if isinstance(evaluator, CLIPEvaluator) or isinstance(evaluator, VisualBertEvaluator):
                evaluator.set_class_names(classes)

            start_time = time.time()
            results[name] = evaluator.evaluate(prompt=sample_prompt, image_path=sample_image_url)
            end_time = time.time()
            print(f"{name} Output: {results[name]} (Evaluated in {end_time - start_time:.2f} seconds)")

        plotting.show_prediction_result(sample_image_url, dataset_name, class_name, results)


def get_class_files(dataset_name, class_name):

    dataset_path = os.path.join("datasets", dataset_name)

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
        raise ValueError(f"Class {class_name} not found in {dataset_name} dataset.")

    class_subfolders = os.listdir(class_path)

    # if folder contains test and train folders, only use test
    if "test" in class_subfolders:
        class_path = os.path.join(class_path, "test")

    # get full file path to the images inside class_path
    files = [os.path.join(class_path, f) for f in os.listdir(class_path) if not f.startswith(".")]

    return files


def get_dataset_classes(name):
    if name == "MVTec_AD":
        # get the folder names inside the MVTec_AD dataset
        # folder = os.listdir("datasets/MVTec_AD") # TODO: Implement this
        return None

    elif name == "BloodMNIST":
        # get the folder names inside the BloodMNIST dataset
        # folder = os.listdir("datasets/BloodMNIST") # TODO: Implement this
        return None

    elif name == "TissueMNIST":
        # get the folder names inside the TissueMNIST dataset
        # folder = os.listdir("datasets/TissueMNIST") # TODO: Implement this
        return None

    elif name == "BreakHis":
        # get the folder names inside the
        # folder = os.listdir("datasets/BreakHis") # TODO: Implement this
        return None

    elif name == "PlantVillage":
        # get the folder names inside the PlantVillage dataset
        folder = os.listdir("datasets/PlantVillage")

    elif name == "FER2013":
        # get the folder names inside the FER2013 dataset
        folder = os.listdir("datasets/FER2013")

    elif name == "EuroSAT":
        # get the folder names inside the EuroSAT dataset
        folder = os.listdir("datasets/EuroSAT")

    elif name == "Food-101":
        # get the folder names inside the Food-101 dataset
        folder = os.listdir("datasets/Food-101")

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # if folder contains test and train folders, only use test
    if "test" in folder:
        folder = os.listdir(f"datasets/{name}/test")

    # if folder contains images, only use images
    if "images" in folder:
        folder = os.listdir(f"datasets/{name}/images")

    # filter out the folders that are not classes or are not visible
    classes = [f for f in folder if not f.startswith(".")]

    return classes




