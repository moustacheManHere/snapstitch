from snapstitch.data import DataLoaderCache, SUPPORTED_FORMATS
import numpy as np


# Using Mocker to simulate the behavior of the glob.glob function
def test_dataloadercache_initialization(mocker):
    def glob_side_effect(pattern, recursive=True):
        # Get the extension of the pattern asked by the DataLoaderCache
        pattern_ext = pattern.split(".")[-1]
        if pattern_ext in SUPPORTED_FORMATS:
            return [f"mock_dir/image1.{pattern_ext}"]
        return []

    mocker.patch("glob.glob", side_effect=glob_side_effect)
    loader = DataLoaderCache(
        image_directory="mock_dir", target_size=(100, 100), max_cache_size=20
    )

    # Check if the number of images loaded is equal to the number of supported formats
    assert len(loader.images) == len(SUPPORTED_FORMATS), (
        f"Expected {len(SUPPORTED_FORMATS)} images, " f"but got {len(loader.images)}"
    )

    # Check that each image corresponds to a file with a supported format
    for image in loader.images:
        file_extension = image.split(".")[-1]
        assert (
            file_extension in SUPPORTED_FORMATS
        ), f"Found unsupported file format {file_extension}"


# Check that the DataLoaderCache can load images from the directory and cache them
def test_dataloadercache_functionality(mocker):
    # harcode the image paths since we aren't testing the glob function
    mocker.patch(
        "glob.glob",
        return_value=[
            "mock_dir/image1.jpg",
            "mock_dir/image2.jpg",
            "mock_dir/image3.jpg",
        ],
    )

    # Mock the cv2.imread function to return a dummy image
    mock_cv2_imread = mocker.patch(
        "cv2.imread", return_value=np.zeros((200, 200, 3), dtype=np.uint8)
    )

    loader = DataLoaderCache(
        image_directory="mock_dir", target_size=(100, 100), max_cache_size=20
    )

    random_images = loader.get_random_images(2)

    # its <= 2 and not == 2 because the image might be loaded from the cache
    assert mock_cv2_imread.call_count <= 2, "Expected 2 calls to cv2.imread only"
    assert len(random_images) == 2, "Expected 1 random image to be returned"
    assert type(random_images[0]) == np.ndarray, "Expected image to be a numpy array"
    assert random_images[0].shape == (100, 100, 3), "Expected image to be resized"
    assert len(loader.cache) <= 2, "Expected 2 images to be cached"
