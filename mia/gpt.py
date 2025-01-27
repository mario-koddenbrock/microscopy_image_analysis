import os

import prompts

# load huggingface access token
cache_path = os.path.expanduser("~/.cache/openai/token")
if os.path.exists(cache_path):
    with open(cache_path, "r") as f:
        API_KEY = f.read().strip()
        print(f"API Key: {API_KEY}")


def get_gpt_response(prompt, openai=None):
    openai = openai.OpenAIApi(API_KEY)
    response = openai.complete(prompt)
    return response.choices[0].text.strip()


def extract_python_code(response):
    code = response.split("```python")[1].split("```")[0].strip()
    return code


def get_augmentation_code(dataset_title, dataset_description):
    prompt = prompts.augmentation_prompt(dataset_title, dataset_description)
    response = get_gpt_response(prompt)
    code = extract_python_code(response)
    return code


def save_code_to_file(code, dataset_title):

    augmentation_path = "augmentations/"
    if not os.path.exists(augmentation_path):
        os.makedirs(augmentation_path)

    file_name = f"{dataset_title}_augmentation.py"
    with open(file_name, "w") as file:
        file.write(code)
