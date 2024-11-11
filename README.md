# Microscopy Image Analysis

This is a project for image analysis in microscopy. The project is part of the DFG Project **Appl-FM**.

**Note:** This repository is currently not operational and should be considered a playground. Implementation is still in progress.

## Overview

The goal of this project is to develop algorithms and tools for automated analysis of microscopic images. This includes functions like image preprocessing, segmentation, feature extraction, and data visualization.

## Installation

All required packages can be installed using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

A first draft is available at `classification.ipynb`. Please note: this is a work in progress.

## Datasets

Some datasets can be automatically downloaded using the `datasets.py` script, while others need to be manually downloaded.

| Type                  | Name                                  | Explanation                                                      | Link                                                                                           |
|-----------------------|---------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| Detection             | AGAR                                  | Microbial colonies on agar plates.                              | [AGAR Dataset](https://agar.neurosys.com/)                                                     |
| Detection             | BacXeption                            | Dataset for bacterial detection.                                 | [BacXeption Dataset](https://github.com/universvm/BacXeption)                                  |
| Classification        | DIBaS                                 | Dataset for bacterial classification.                            | [DIBaS Dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC5599001/)                             |
| Classification        | Capstone Project                      | Dataset for bacterial classification from a capstone project.    | [Capstone Project Dataset](https://github.com/deibyrios/bacteria-classification)               |

## Projects

| Type                  | Name                                  | Explanation                                                      | Link                                                                                           |
|-----------------------|---------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| AMR Detection         | Bacteria Detection for AMR            | Detecting antimicrobial resistance (AMR) in bacteria.            | [Bacteria Detection Project](https://github.com/biodatlab/bacteria-detection/tree/master)      |
| 3D Segmentation       | 3DCellSeg                             | 3D cell segmentation using deep learning.                        | [3DCellSeg Project](https://github.com/AntonotnaWang/3DCellSeg)                                |
| 3D Segmentation       | StarDist                              | Star-convex shape models for 3D microscopy images.               | [StarDist Project](https://github.com/stardist/stardist)                                       |
| 2D Segmentation       | Automatic SEM Image Segmentation      | 2D segmentation of SEM images.                                   | [Automatic SEM Image Segmentation Project](https://github.com/BAMresearch/automatic-sem-image-segmentation) |

## Contributing

Contributions are welcome! If you have ideas or suggestions for improvement, please open an issue or a pull request.

## Contact

For questions, please contact Mario Koddenbrock at [mario.koddenbrock@htw-berlin.de](mailto:mario.koddenbrock@htw-berlin.de).

