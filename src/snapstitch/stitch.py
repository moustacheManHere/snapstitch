from typing import Dict
from .data import PartsLoader, BackgroundLoader
from .generator import Generator


# Main Stitching Logic
class Stitcher:
    def __init__(
        self,
        generator: Generator,
        background: BackgroundLoader,
        parts: Dict[str, PartsLoader],
        parts_per_image: int = 30,
    ) -> None:
        self.generator = generator
        self.background = background
        self.parts = parts
        self.parts_per_image = parts_per_image

    def execute(
        self, num_images: int, output_folder: str
    ) -> None:  # Later add more parameters
        pass
