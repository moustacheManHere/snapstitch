from typing import Dict, Tuple
from .data import PartsLoader, BackgroundLoader
from .generator import Generator
from tqdm import tqdm
import logging
import random

# Create type with PartsLoader and scale
PartsLoaderWithScale = Tuple[PartsLoader, float]


# Main Stitching Logic
class Stitcher:
    def __init__(
        self,
        generator: Generator,
        background: BackgroundLoader,
        parts: Dict[str, PartsLoaderWithScale],  # for negative samples, make the key '_'
        parts_per_image: int = 30,
    ) -> None:
        self.generator = generator
        self.background = background

        # for parts loop thru the keys and get the parts loader
        self.parts = {}
        self.classes = []
        self.proportions = []

        for class_name, parts_loader in parts.items():
            # Ensure the value is a tuple with the correct structure
            if len(parts_loader) != 2:
                raise ValueError(f"Value for key {class_name} must be a tuple of (PartsLoader, float).")

            self.parts[class_name] = parts_loader[0]
            self.proportions.append(parts_loader[1])
            self.classes.append(class_name)

        self.parts_per_image = parts_per_image

        if self.classes is None:
            raise ValueError("Classes must be provided")

    def execute(
        self,
        num_images: int,
        output_folder: str,
        image_name: str,
        train_or_val: bool = True,
        perimeter_start: Tuple[int, int] = (0, 0),
        perimeter_end: Tuple[int, int] = (2560, 1440),
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
                class_name = random.choices(self.classes, weights=self.proportions, k=1)[0]
                parts_loader = self.parts[class_name]
                if class_name == '_':
                    class_id = -1
                else:
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
                perimeter_start=perimeter_start,
                perimeter_end=perimeter_end
            )

            if not success:
                logging.warning("Failed to generate image")
                return

        logging.info(f"Generated image {image_num+1}/{num_images}")
        return
