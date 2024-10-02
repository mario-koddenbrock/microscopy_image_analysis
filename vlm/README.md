# VLM Package

The VLM (Visual Learning Model) package is designed for evaluating and classifying visual learning models. It provides utilities for initializing evaluators, evaluating individual images and datasets, and managing dataset files.

## Features

- **Initialize Evaluators**: Set up evaluators for image analysis.
- **Evaluate Image**: Analyze a single image using the initialized evaluators.
- **Evaluate Dataset**: Analyze a dataset of images and generate results.
- **Manage Dataset Files**: Retrieve class files and dataset classes.


## Usage
### Initialize Evaluators
```python
from vlm.utils import initialize_evaluators

config = {'evaluator1': 'config1', 'evaluator2': 'config2'}
device = 'cpu'
evaluators = initialize_evaluators(device, config)
```

### Evaluate Image
```python
from vlm.utils import evaluate_image

evaluators = {'evaluator1': 'config1'}
image_path = 'path/to/image.jpg'
result = evaluate_image(evaluators, image_path)
print(result)
```

### Evaluate Dataset
```python
from vlm.utils import evaluate_dataset

evaluators = {'evaluator1': 'config1'}
dataset_name = 'dataset'
dataset_description = 'description'
dataset_path = 'path/to/dataset'
num_images_per_class = 1
num_classes = 33
results = evaluate_dataset(evaluators, dataset_name, dataset_description, dataset_path, num_images_per_class, num_classes)
print(results)
```


### Get Class Files
```python
from vlm.utils import get_class_files

dataset_path = 'path/to/dataset'
class_name = 'class1'
files = get_class_files(dataset_path, class_name)
print(files)
```

### Get Dataset Classes
```python
from vlm.utils import get_dataset_classes

dataset_path = 'path/to/dataset'
classes = get_dataset_classes(dataset_path)
print(classes)
```



