# SnapStitch

SnapStitch is an open-source Python library for generating synthetic datasets to enhance computer vision tasks. It simplifies the creation of superimposed images, making it especially useful when working with limited datasets.

## How It Works

Image stitching is a process that generates synthetic data by placing smaller image components onto a larger background canvas. This method helps create a diverse dataset, ideal for training object detection models. Below is a step-by-step breakdown:

### Example: Circuit Board Dataset

To train a model to detect components on a circuit board, you typically need a large variety of images showing the board in different orientations and configurations.

<img src="docs/circuit_board.png" alt="Circuit Board" style="width: 50%; display: block; margin: auto; padding: 1rem 0;"/>

Instead we propose the following approach:
1. Take a picture of a circuit board.
2. Extract parts like resistors, capacitors, or chips from the image.
3. Use an empty or simplified circuit board as the background.
4. Place the cropped components on the canvas in different positions and orientations.

<img src="docs/stitching_process.png" alt="Circuit Board" style="width: 80%; display: block; margin: auto;padding: 1rem 0;"/>


This approach can:
- Quickly generate a large and varied dataset.
- Introduce variability in positions, angles, and layouts.
- Reduces overfitting by providing more diverse training data.

## Usage

To install the `snapstitch` package, run:

```bash
pip install snapstitch
```

Assuming you have a folder structure like this:

```
snapstitch_project/
│
├── supermarket_dataset/
│   ├── background/
│   │   └── background_image.jpg
│   │   └── ...
│   ├── parts/
│   │   ├── bread/
│   │   │   ├── bread_part1.png
│   │   │   ├── bread_part2.png
│   │   │   └── ...
│   │   ├── canned_beans/
│   │   │   ├── canned_beans_part1.png
│   │   │   ├── canned_beans_part2.png
│   │   │   └── ...
│   │   └── jam/
│   │       ├── jam_part1.png
│   │       ├── jam_part2.png
│   │       └── ...
│   └── output/
│
└── requirements.txt
```

You can use the following code to generate a synthetic YOLO dataset:

```python
from snapstitch import Stitcher, PartsLoader, BackgroundLoader, YOLOv8Generator

# Initialise path to backgrounds
background = BackgroundLoader("supermarket_dataset/background")

# Initialise all your classes
bread = PartsLoader("supermarket_dataset/parts/bread")
canned_beans = PartsLoader("supermarket_dataset/parts/canned_beans")
jam = PartsLoader("supermarket_dataset/parts/jam")

# Generate YOLOv8 data 
generator = YOLOv8Generator()

# Main class that handles generation
stitcher = Stitcher(generator, background, {"bread": bread, "canned_beans": canned_beans, "jam":jam}, 30, ["bread", "canned_beans", "jam"])

# Generate as many times as needed
stitcher.execute(10, "supermarket_dataset/output", "train_data1")
stitcher.execute(10, "supermarket_dataset/output", "train_data2")
stitcher.execute(10, "supermarket_dataset/output", "train_data3")
```

For a YOLOv8 Generator, the output will be organized as follows:
```
snapstitch_project/
│
├── supermarket_dataset/
│   ├── background/
│   ├── parts/
│   │   ├── bread/
│   │   ├── canned_beans/
│   │   └── jam/
│   └── output/
│       ├── images/
│       │   ├── train/
│       │   │   ├── train_data1_0.png
│       │   │   ├── train_data1_1.png
│       │   │   └── ...
│       │   └── val/
│       │       ├── val_data1_0.png
│       │       ├── val_data1_1.png
│       │       └── ...
│       └── labels/
│           ├── train/
│           │   ├── train_data1_0.txt
│           │   ├── train_data1_1.txt
│           │   └── ...
│           └── val/
│               ├── val_data1_0.txt
│               ├── val_data1_1.txt
│               └── ...
│
└── requirements.txt
```

## Features

This library is currently under development. Key features and functionalities are being implemented step-by-step. The following features have been implemented so far:

- **Superimpose objects onto backgrounds** to quickly generate diverse, labeled datasets for object detection tasks.
- **Automated YOLOv8 dataset generation**, including images and labels in the required format for immediate model training.

Aiming to give users more control and extend this to other computer vision tasks, we aim to:
- **Support for additional annotation formats**: Expand compatibility to other object detection frameworks like Pascal VOC and COCO.
- **Advanced augmentation techniques**: Allow users to add augmentations to the images before stitching through the Albumentations library.
- **Other Computer Vision tasks**: Generate synthetic datasets for segmentation, oriented-bounding boxes, and pose estimation.

## Contributing

Contributions to SnapStitch are welcome! Here’s how you can help:

1. **Fork the Repository**: Click on the fork button in the upper right corner of the GitHub repository.

2. **Clone Your Fork**: Clone your fork to your local machine using:
   ```bash
   git clone https://github.com/yourusername/snapstitch.git
   ```

3. **Set Up Your Development Environment**:
   - Navigate into the cloned directory:
     ```bash
     cd snapstitch
     ```
   - Initialize PDM for the project:
     ```bash
     pdm install
     ```

4. **Create a Branch**: Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make Changes**: Implement your feature or fix the bug.

6. **Run Tests**: Before committing, run the tests to ensure everything is working correctly:
   ```bash
   pdm run pytest
   ```

7. **Check Flake8 Compliance**: Ensure your code complies with the PEP 8 style guide by running Flake8:
   ```bash
   pdm run flake8
   ```
   Address any issues reported by Flake8 before proceeding.

8. **Commit Your Changes**: Commit your changes with a clear and concise message:
   ```bash
   git commit -m "Add feature: your feature description"
   ```

9. **Push to Your Fork**: Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

10. **Submit a Pull Request**: Go to the original repository and submit a pull request with a description of your changes.

## License

SnapStitch is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

