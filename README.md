# General ML Framework

General ML Framework is a repository that provides a comprehensive set of tools and utilities for training machine learning models for image classification, image segmentation, and object detection tasks. The repository is designed to be easily modifiable and extensible, allowing users to incorporate new datasets, models, losses, and more.

## Features

- **Image Classification**: Train models to classify images into multiple classes.
- **Image Segmentation**: Train models to perform pixel-level segmentation of images.
- **Object Detection**: Train models to detect and localize objects within images.
- **Easy Customization**: Easily incorporate new datasets, models, losses, and other components.
- **Flexible Configuration**: Configure training parameters, hyperparameters, and experiment settings.
- **Preprocessing and Augmentation**: Apply various preprocessing techniques and data augmentation strategies.
- **Evaluation and Metrics**: Evaluate model performance using standard metrics for each task.
- **Visualization**: Visualize model predictions, ground truth annotations, and other analysis outputs.
- **Checkpointing and Logging**: Save checkpoints of trained models and log training progress.

## Installation

1. Clone the repository:
```
git clone https://github.com/your-username/ml-image-toolkit.git
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Set up the project environment:

Optional: Create a virtual environment
```
python -m venv env
source env/bin/activate # Activate the virtual environment (Linux/macOS)
```
OR
```
.\env\Scripts\activate # Activate the virtual environment (Windows)
```

## Usage

1. Prepare your dataset:

- Basic datasets are provided for training a basic model
- Write custom dataset class to fit your specific dataset

2. Configure the training settings:

- A bunch of parameters related to training, optimization, etc. are possible to tune in *TASK*/main.py

3. Start the training:

- Run the training script for your desired task:

  Go to *TASK*/main.py and adjust the if-statements according to your desire

4. Monitor the training:

- View training logs, loss curves, and other visualizations that are generated during training

5. Evaluate the trained models:

- Run the evaluation script for your desired task:

  Go to *TASK*/main.py and adjust the if-statements according to your desire

6. Customize and extend:

- Explore the repository structure and modify the code to suit your specific needs.
- Add new datasets, models, losses, or other components using the provided templates and guidelines.

## Contributing

Contributions to General ML Framework are welcome! If you encounter any issues, have suggestions, or would like to contribute new features or improvements, please open an issue or submit a pull request.

