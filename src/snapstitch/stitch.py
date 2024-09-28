from typing import Dict, List
from .data import PartsLoader, BackgroundLoader
from .generator import Generator
from tqdm import tqdm
import logging
import random


# Main Stitching Logic
class Stitcher:
    def __init__(
        self,
        generator: Generator,
        background: BackgroundLoader,
        parts: Dict[str, PartsLoader],
        parts_per_image: int = 30,
        classes: List[str] = None,
    ) -> None:
        self.generator = generator
        self.background = background
        self.parts = parts
        self.parts_per_image = parts_per_image
        self.classes = classes
        if self.classes is None:
            raise ValueError("Classes must be provided")

        if len(self.parts) != len(self.classes):
            raise ValueError("Number of classes must match the number of parts loaders")

    def execute(
        self,
        num_images: int,
        output_folder: str,
        image_name: str,
        train_or_val: bool = True,
    ) -> None:  # Later add more parameters

        for image_num in tqdm(range(num_images)):
            # Get a random background image
            background_image = self.background.get_random_images()[0]
            if background_image is None:
                logging.warning("No background image found, exiting")
                return

            # Get random parts for the image
            parts = []
            classes = []

            for i in range(self.parts_per_image):
                # Get a random class
                class_name = random.choice(self.classes)
                parts_loader = self.parts[class_name]
                class_id = self.classes.index(class_name)

                part = parts_loader.get_random_images()[0]
                if part is None:
                    logging.warning(f"No part found for class {class_name}")
                    continue

                parts.append(part)
                classes.append(class_id)

            # Generate the image
            success = self.generator.generate(
                background_image,
                parts,
                classes,
                output_folder,
                f"{image_name}_{image_num}",
                train_or_val,
            )

            if not success:
                logging.warning("Failed to generate image")
                return

        logging.info(f"Generated image {image_num+1}/{num_images}")
        return
