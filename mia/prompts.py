from mia import utils


def classification_prompt(
    dataset_name: str = "Bacterial Species",
    dataset_description: str = "Digital Image of Bacterial Species",
    dataset_path: str = "datasets/Classification",
):
    class_names = utils.get_dataset_classes(dataset_path)

    prompt = f"""
I have a dataset titled '{dataset_name}' with the following description: 
{dataset_description}

The dataset consists of images of the following classes: {class_names}.

Here is a sample image from the dataset. Classify the image into one of the classes.
Only return the class label.
            """

    # save the prompt to a file
    with open(f"prompts/{dataset_name}_classification_prompt.txt", "w") as file:
        file.write(prompt)

    return prompt


if __name__ == "__main__":

    dataset_name = "Digital Image of Bacterial Species"
    dataset_description = """
    The dataset from the study "Deep learning approach to bacterial colony classification" comprises 660 images representing 33 different genera and species of bacteria. 
    This dataset, called DIBaS (Digital Image of Bacterial Species), was created for bacterial classification using deep learning methods. 
    The images were taken with a microscope and analyzed using Convolutional Neural Networks (CNNs) and machine learning classifiers like Support Vector Machines (SVM) and Random Forest. 
    The dataset is publicly available for research purposes, allowing for advancements in bacterial recognition systems.
    """
    dataset_path = "../datasets/Classification"
    num_images = 1

    prompt = classification_prompt(dataset_name, dataset_description, dataset_path)

    print(f"{dataset_name}:\n")
    print(prompt)
