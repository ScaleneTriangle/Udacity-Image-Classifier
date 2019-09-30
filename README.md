# Udacity Image Classifier

Image classifier project from Udacity Data Scientist Course.

Image classifier that uses the pytorch pretrained models as a feature extractor then trains a classifier based on a provided set of images (flowers were used in this example).

Flower dataset is not included in this repository.

Contents:
- Image Classifier Project.ipynb: Modifications to the Udacity provided Jupyter Notebook file
- model.py: Code containing model architecture, save, and load functions
- predict.py: Code to call the trained model checkpoint
- train.py: Code to train the architecture from model.py
- utility_functions: Code containing data loaders, image processors, and result output
- workspace_utils.py: provided tools for Udacity Workspace
- cat_to_name.json: Provide file containing names for assets
- assets: folder containing images for the Jupyter Notebook
