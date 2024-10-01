import datasets


def augmentation_prompt(
        dataset_title:str = "forestry",
        dataset_description:str = "drone images of forests",
):

    prompt = f"""
I have a dataset titled '{dataset_title}' with the following description: '{dataset_description}'. 

The dataset consists of images, and I need to apply relevant data perturbation techniques to simulate 
the kinds of variations and distortions that commonly occur in real-world scenarios when handling 
this specific type of imagery.

Generate a Python function that includes suitable perturbation methods specific to the dataset's context. 
These perturbations should reflect realistic challenges associated with the domain, such as noise from sensors, 
lighting variations, changes in perspective or focus, or distortions caused by environmental factors relevant 
to the dataset. The perturbations should be tailored to the nature of the data; for instance, if the images 
are aerial or remote sensing data, include transformations such as wind-induced movement or perspective shifts, 
while microscopy data might require perturbations related to sensor noise, focus variations, or illumination changes.

The Python function should meet the following requirements:
1. It should take an image as input and accept optional parameters for each perturbation method 
   (e.g., degree of rotation, noise intensity, color adjustment levels).
2. The function should include a diverse set of perturbations that are contextually relevant, such as random 
   rotations, flips, scaling, noise addition, perspective transformations, or domain-specific challenges 
   (e.g., lighting variations, focus shifts).
3. The function should return the perturbed image as the output.
4. The function should utilize libraries such as `opencv`, `numpy`, or other suitable libraries for 
   performing these image perturbations.
            """

    return prompt


def classification_prompt(
        dataset_name:str = "forestry",
        dataset_description:str = "drone images of forests",
):
    dataset_names = datasets.get_dataset_classes(dataset_name)

    prompt = f"""
I have a dataset titled '{dataset_name}' with the following description: '{dataset_description}'.
The dataset consists of images of the following classes: {dataset_names}.

Here is a sample image from the dataset. Classify the image into one of the classes.
Only return the class label.
            """

    return prompt



if __name__ == "__main__":

    datasets = datasets.DATASET_DESCRIPTIONS

    # iterate over the datasets and generate augmentation prompts
    for dataset in datasets:
        dataset_name = dataset
        dataset_description = datasets[dataset]
        prompt = augmentation_prompt(dataset_name, dataset_description)

        # save the prompt to a file
        with open(f"prompts/{dataset_name}_augmentation_prompt.txt", "w") as file:
            file.write(prompt)

        print(f"{dataset}: {dataset_description}")
        print("\n\n")
        print(prompt)
        print("\n\n")
        print("-------------------------------------------------------------------------------------------------------------------")


