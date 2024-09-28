# Standard Library Imports
import os
import glob
import logging
import random
from collections import OrderedDict
from typing import List, Tuple, Optional

# Third Party Imports
import cv2
import numpy as np

# Constants
SUPPORTED_FORMATS = ["png", "jpg", "jpeg"]

# Types for type hinting
ImageSize = Tuple[int, int]


# Base Data Class
class DataLoaderCache:
    def __init__(
        self, image_directory: str, target_size: ImageSize, max_cache_size: 20
    ) -> None:

        # Initialize the cache to store frequently used images
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size

        # Initialize the images directory
        self.image_directory = image_directory
        self.images = []

        # Get all images from the directory
        self._get_backgrounds_from_directory(self.image_directory)
        if not self.images:
            raise ValueError(
                "No images found in the directory: {}".format(image_directory)
            )

        # Initialize the target size
        self.target_size = target_size

    def _get_backgrounds_from_directory(self, directory_path: str) -> None:
        # Create a pattern to match all supported formats in subdirectories
        for ext in SUPPORTED_FORMATS:
            pattern = os.path.join(directory_path, "**", f"*.{ext}")
            self.images.extend(glob.glob(pattern, recursive=True))

        return

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        try:
            # Load the image using OpenCV
            image = cv2.imread(image_path)
        except Exception as e:
            self.images.remove(image_path)
            logging.error("Error loading image: {}".format(image_path))
            logging.error("Error: {}".format(e))
            return None

        # Resize the image
        image = self._resize_image(image)
        if image is None:
            logging.error("Error resizing image: {}".format(image_path))
            return None

        return image

    # Making this a separate function to overidde in the PartsLoader class
    def _resize_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            image = cv2.resize(image, self.target_size)
        except Exception as e:
            logging.error("Error: {}".format(e))
            return None
        return image

    def _cache_image(self, file_path: str, image) -> None:

        if len(self.cache) >= self.max_cache_size:
            # Remove the least recently used image
            self.cache.popitem(last=False)
        self.cache[file_path] = image

    def _get_random_image(self) -> Optional[np.ndarray]:

        # Check if the images are empty
        if len(self.images) == 0:
            return None

        # Get a random image from the list of images
        image_path = random.choice(self.images)

        if image_path in self.cache:
            # Move the image to the end of the cache
            image = self.cache.pop(image_path)
            self.cache[image_path] = image
        else:
            # Load the image
            image = self._load_image(image_path)
            if image is None:
                return None
            # Cache the image
            self._cache_image(image_path, image)

        return image

    def get_random_images(self, num_images: int = 1) -> List[np.ndarray]:
        images = []

        while len(images) < num_images:
            image = self._get_random_image()
            if image is not None:
                images.append(image)

        return images


# Background Loader
class BackgroundLoader(DataLoaderCache):
    def __init__(
        self,
        image_dir: str,
        target_size: ImageSize = (2560, 1440),
        max_cache_size: int = 20,
    ) -> None:
        super().__init__(image_dir, target_size, max_cache_size)


# Parts Loader
class PartsLoader(DataLoaderCache):
    def __init__(
        self,
        image_directory: str,
        target_size: ImageSize = (400, 400),
        scale: float = 1,
        max_cache_size: int = 20,
    ) -> None:
        self.scale = scale
        target_size = (int(target_size[0] * scale), int(target_size[1] * scale))
        super().__init__(image_directory, target_size, max_cache_size)

    def _resize_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Target size will be something like [300,300]
        but we need to account for the aspect ratio of parts.
        So I will override this function.

        Also we need to scale the parts by the scale factor for that class.
        """

        height, width = image.shape[:2]
        aspect_ratio = width / height

        target_width, target_height = self.target_size

        if aspect_ratio > target_width / target_height:
            # Resize based on width
            new_width = target_width
            new_height = int(new_width / aspect_ratio)
        else:
            # Resize based on height
            new_height = target_height
            new_width = int(new_height * aspect_ratio)

        try:
            image = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
        except Exception as e:
            logging.error("Error: {}".format(e))
            return None
        return image
