from typing import List, Tuple, Optional
import numpy as np
import random
import os
import logging

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
        image_dir: str,
        label_dir: str,
        image_name: str,
    ) -> bool:  # Return indicates success or failure

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        if len(parts) == 0 or len(classes) == 0 or len(parts) != len(classes):
            return False

        current_positions = []  # Array to hold all current part positions
        for part, class_id in zip(parts, classes):
            part_size = part.shape[:2]
            new_position = self._get_new_part_position(
                current_positions, part_size, background.shape[:2]
            )

            if new_position is None:
                continue

            new_position = new_position + (class_id,)

            current_positions.append(new_position)
            background = self._place_part(background, part, new_position)

        # Save the data to disk
        image_path = os.path.join(image_dir, image_name + ".jpg")
        success = self._save_image(background, image_path)
        if not success:
            return False

        label_path = os.path.join(label_dir, image_name + ".txt")
        self._save_labels(current_positions, label_path)
        if not success:
            os.remove(image_path)  # Remove the image if labels are not saved
            return False

        pass

    def _get_new_part_position(
        current_positions: List[PartPlacement],
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
                current_x1, current_y1 = current_position[0]
                current_x2, current_y2 = current_position[1]

                if (
                    random_x < current_x2
                    and random_y < current_y2
                    and random_x + part_size[0] > current_x1
                    and random_y + part_size[1] > current_y1
                ):
                    overlap = True
                    break

            if overlap:
                continue

            return (random_x, random_y), (
                random_x + part_size[0],
                random_y + part_size[1],
            )

        return None

    def _place_part(
        background: np.ndarray, part: np.ndarray, position: PartPlacement
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
        bg_slice = background[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
        part_slice = part[
            : y2_clamped - y1_clamped, : x2_clamped - x1_clamped
        ]  # Sliced part fitting the background
        alpha_slice = part_alpha[: y2_clamped - y1_clamped, : x2_clamped - x1_clamped]

        # Add the part to the background
        background[y1_clamped:y2_clamped, x1_clamped:x2_clamped] = (
            1 - alpha_slice[:, :, np.newaxis]
        ) * bg_slice + alpha_slice[:, :, np.newaxis] * part_slice

    def _save_labels(
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

    def _save_image(image: np.ndarray, output_path: str) -> bool:
        try:
            cv2.imwrite(output_path, image)
            return True
        except Exception as e:
            logging.error(f"Error saving image: {e}")
            return False
