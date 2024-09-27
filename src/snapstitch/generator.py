from typing import List, Tuple, Optional
import numpy as np
import random

# Types for type hinting [x,y]
Coordinates = Tuple[int, int]


# Generic Class for all generators
class Generator:
    def __init__(self) -> None:
        pass

    # Any generator must implement the generate method to be used by stitcher
    def generate(
        self,
        background: np.ndarray,
        parts: List[np.ndarray],
        image_dir: str,
        label_dir: str,
        image_name: str,
    ) -> List[np.ndarray]:
        raise NotImplementedError("Use a particular generator class")


# YOLOv8 Generator
class YOLOv8Generator(Generator):
    def __init__(self, overlap_ratio=0.1) -> None:
        super().__init__()
        self.overlap_ratio = overlap_ratio

    def generate(
        self,
        background: np.ndarray,
        parts: List[np.ndarray],
        image_dir: str,
        label_dir: str,
        image_name: str,
    ) -> List[np.ndarray]:

        # Loop through all the parts
        # Get a random position for each
        # Place the part on the background
        # Save the labels
        pass

    def _get_new_part_position(
        current_positions: List[Coordinates],
        part_size: Coordinates,
        background_size: Coordinates,
    ) -> Optional[Coordinates]:
        max_attempts = 10

        # try to find a random position that doesn't overlap
        for i in range(max_attempts):
            # x,y refers to the top left coordinate of the part, not center
            random_x = random.randint(0, background_size[0] - part_size[0])
            random_y = random.randint(0, background_size[1] - part_size[1])

            # Check if the new part overlaps with any existing parts
            overlap = False
            for current_position in current_positions:
                # TODO: Check if the overlap ratio is within the threshold
                if (
                    random_x < current_position[0] + part_size[0]
                    and random_x + part_size[0] > current_position[0]
                    and random_y < current_position[1] + part_size[1]
                    and random_y + part_size[1] > current_position[1]
                ):
                    overlap = True
                    break

            if overlap:
                continue

            return random_x, random_y

        return None

    def _place_part(
        background: np.ndarray, part: np.ndarray, position: Coordinates
    ) -> np.ndarray:
        pass

    def _save_labels(current_positions: List[Coordinates], output_path: str) -> bool:
        # need to convert x1,y1,x2,y2 to x_center, y_center, width, height for YOLO
        pass

    def _save_image(image: np.ndarray, output_path: str) -> bool:
        pass
