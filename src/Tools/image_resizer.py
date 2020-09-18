# Example usage
# python3 src/Tools/image_resizer.py\
# --base_path=data\
# --resolution=400x400

import logging
import argparse
import pathlib
import sys
from PIL import Image

LOGGER_NAME = "ImageResizer"


def __parse_width_height(opt: argparse.Namespace):
    width, height = opt.resolution.split("x")

    return int(width), int(height)


def __infer_image_directory(base_path: pathlib.Path):
    logger = logging.getLogger(LOGGER_NAME)

    def __contains_image(path: pathlib.Path):
        for path in path.glob("*"):
            if path.is_file() and str(path).split(".")[-1] == "png":
                return True
        return False

    for path in base_path.glob("*"):
        if path.is_dir() and __contains_image(path):
            logger.info(f"Inferred image directory to be {path}")
            return path

    raise FileNotFoundError("Did not find image directory")


def __add_converted_images(width: int,
                           height: int,
                           image_directory: pathlib.Path):
    logger = logging.getLogger(LOGGER_NAME)
    for image_path in image_directory.glob("*"):
        logger.info(f"converting {image_path} to {width}x{height}")
        with open(image_path, "rb") as image_file:
            image = Image.open(image_file).resize((width, height))

        with open(str(image_path) + f".{width}x{height}", "wb") as image_file:
            image.save(image_file, format="png")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(LOGGER_NAME)

    parser = argparse.ArgumentParser(
        description="This is a tool for renaming the open-review images to a different size")
    parser.add_argument("--base_path",
                        help="base path of dataset",
                        required=True,
                        type=str)
    parser.add_argument("--resolution",
                        help="resolution on format YYYYxZZZZ e.g 1980x1240",
                        required=True,
                        type=str)

    args = parser.parse_args()
    width, height = __parse_width_height(args)

    base_path = pathlib.Path(args.base_path)

    try:
        image_directory = __infer_image_directory(base_path)
        __add_converted_images(width=width, height=height, image_directory=image_directory)
    except FileNotFoundError:
        logger.fatal(f"Could not infer image directory, no subdirectory contained png files")
