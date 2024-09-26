from typing import List, Tuple
import numpy as np


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
        pass

    def _get_new_part_position(
        current_positions: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        pass

    def _place_part(
        background: np.ndarray, part: np.ndarray, position: Tuple[int, int]
    ) -> np.ndarray:
        pass

    def _save_labels(
        current_positions: List[Tuple[int, int]], output_path: str
    ) -> bool:
        pass

    def _save_image(image: np.ndarray, output_path: str) -> bool:
        pass
