from typing import List, Tuple, Optional
import numpy as np
import random
import os
import logging
import cv2

# Types for type hinting [x,y]
Coordinates = Tuple[int, int]
PartPlacement = Tuple[Coordinates, Coordinates, int]  # (top_left, bottom_right, class)


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
        classes: List[int],
        output_dir: str,
        image_name: str,
        train_or_val: bool,  # True for train, False for val
    ) -> bool:  # Return indicates success or failure

        # join the image dir, train/val, and image name
        if train_or_val:
            image_dir = os.path.join(output_dir, "images", "train")
            label_dir = os.path.join(output_dir, "labels", "train")
        else:
            image_dir = os.path.join(output_dir, "images", "val")
            label_dir = os.path.join(output_dir, "labels", "val")

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        if len(parts) == 0 or len(classes) == 0 or len(parts) != len(classes):
            logging.error("Invalid parts or classes")
            return False

        current_positions = []  # Array to hold all current part positions
        background_copy = background.copy()
        for part, class_id in zip(parts, classes):
            part_size = part.shape[:2]
            new_position = self._get_new_part_position(
                current_positions, part_size, background_copy.shape[:2]
            )

            if new_position is None:
                continue

            new_position = new_position + (class_id,)

            current_positions.append(new_position)
            background_copy = self._place_part(background_copy, part, new_position)

        # Save the data to disk

        image_path = os.path.join(image_dir, image_name + ".jpg")
        success = self._save_image(background_copy, image_path)
        if not success:
            logging.error(f"Error saving image {image_path}")
            return False

        label_path = os.path.join(label_dir, image_name + ".txt")
        background_size = (background_copy.shape[1], background_copy.shape[0])
        self._save_labels(current_positions, label_path, background_size)
        if not success:
            os.remove(image_path)  # Remove the image if labels are not saved
            logging.error(f"Error saving labels {label_path}")
            return False

        return True

    def _get_new_part_position(
        self,
        current_positions: List[PartPlacement],
        part_size: Coordinates,
        background_size: Coordinates,
    ) -> Optional[Coordinates]:
        max_attempts = 10

        # try to find a random position that doesn't overlap
        for i in range(max_attempts):
            # x,y refers to the top left coordinate of the part, not center
            random_x = random.randint(0, background_size[1] - part_size[1])
            random_y = random.randint(0, background_size[0] - part_size[0])

            # Coordinates of the new part (top-left and bottom-right)
            new_x1, new_y1 = random_x, random_y  # top-left corner
            new_x2, new_y2 = (
                random_x + part_size[1],
                random_y + part_size[0],
            )  # bottom-right corner
            part_width, part_height = part_size
            part_area = part_width * part_height

            # Check if the new part overlaps with any existing parts
            overlap = False
            for position in current_positions:
                # check for overlap
                existing_x1, existing_y1 = position[0]
                existing_x2, existing_y2 = position[1]

                # TODO add overlap ratio feature
                dx = min(existing_x2, new_x2) - max(existing_x1, new_x1)
                dy = min(existing_y2, new_y2) - max(existing_y1, new_y1)

                if dx > 0 and dy > 0:  # Only consider positive overlap areas
                    overlap_area = dx * dy
                    overlap_ratio = overlap_area / part_area

                    # If the overlap area is larger than allowed, set overlap to True
                    if overlap_ratio > 0:
                        overlap = True
                        break

            if overlap:
                continue

            return (new_x1, new_y1), (new_x2, new_y2)

        return None

    def _place_part(
        self, background: np.ndarray, part: np.ndarray, position: PartPlacement
    ) -> np.ndarray:
        part_width, part_height = part.shape[1], part.shape[0]
        background_width, background_height = background.shape[1], background.shape[0]

        # If the part is transparent, get the alpha channel
        if part.shape[2] == 4:
            part_alpha = part[:, :, 3] / 255.0
            part = part[:, :, :3]
        else:
            part_alpha = np.ones((part_height, part_width))
        # Get the coordinates to place part
        x1, y1 = position[0]  # Rmb these are the top left coordinates
        x2, y2 = position[1]

        # TODO: check if the clamping affects the alpha channel
        x1_clamped = max(x1, 0)
        x2_clamped = min(x2, background_width)
        y1_clamped = max(y1, 0)
        y2_clamped = min(y2, background_height)

        # Basically get the background and part at the coords
        # Add them using alpha channel to account for transparency
        bg_slice = background[y1_clamped:y2_clamped, x1_clamped:x2_clamped, :]
        part_slice = part[
            : y2_clamped - y1_clamped, : x2_clamped - x1_clamped, :
        ]  # Sliced part fitting the background

        alpha_slice = part_alpha[: y2_clamped - y1_clamped, : x2_clamped - x1_clamped]

        # Add the part to the background
        background[y1_clamped:y2_clamped, x1_clamped:x2_clamped] = (
            1 - alpha_slice[:, :, np.newaxis]
        ) * bg_slice + alpha_slice[:, :, np.newaxis] * part_slice

        return background

    def _save_labels(
        self,
        current_positions: List[Coordinates],
        output_path: str,
        background_size: Coordinates,
    ) -> bool:
        # need to convert x1,y1,x2,y2 to x_center, y_center, width, height for YOLO
        # need to normalize the values by dividing by the background size
        to_write = ""
        for position in current_positions:
            x1, y1 = position[0]
            x2, y2 = position[1]
            class_id = position[2]

            x_center = (x1 + x2) / 2 / background_size[0]
            y_center = (y1 + y2) / 2 / background_size[1]
            width = (x2 - x1) / background_size[0]
            height = (y2 - y1) / background_size[1]

            to_write += f"{class_id} {x_center} {y_center} {width} {height}\n"

        try:
            with open(output_path, "w") as f:
                f.write(to_write)
            return True
        except Exception as e:
            logging.error(f"Error saving labels: {e}")
            return False

    def _save_image(self, image: np.ndarray, output_path: str) -> bool:
        try:
            cv2.imwrite(output_path, image)
            return True
        except Exception as e:
            logging.error(f"Error saving image: {e}")
            return False
