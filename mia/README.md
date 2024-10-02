# MIA Package

The MIA (Microbial Image Analysis) package is designed for evaluating and classifying microbial images. It provides utilities for initializing evaluators, evaluating individual images and datasets, and managing dataset files.

## Features

- **Initialize Evaluators**: Set up evaluators for image analysis.
- **Evaluate Image**: Analyze a single image using the initialized evaluators.
- **Evaluate Dataset**: Analyze a dataset of images and generate results.
- **Manage Dataset Files**: Retrieve class files and dataset classes.


## Usage
### Initialize Evaluators
```python   
from mia.utils import initialize_evaluators

config = {'evaluator1': 'config1', 'evaluator2': 'config2'}
device = 'cpu'
evaluators = initialize_evaluators(device, config)
```

### Evaluate Image
```python   
from mia.utils import evaluate_dataset

evaluators = {'evaluator1': 'config1'}
dataset_name = 'dataset'
dataset_description = 'description'
dataset_path = 'path/to/dataset'
num_images_per_class = 1
num_classes = 2
results = evaluate_dataset(evaluators, dataset_name, dataset_description, dataset_path, num_images_per_class, num_classes)
print(results)
```

### Get Class Files
```python   
from mia.utils import get_class_files

dataset_path = 'path/to/dataset'
class_name = 'class1'
files = get_class_files(dataset_path, class_name)
print(files)
```


