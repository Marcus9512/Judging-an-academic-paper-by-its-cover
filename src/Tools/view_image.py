# Example usage
# python3 src/Tools/view_image.py\
# --npy_path=data/images/xxx.npy\
# --index=0

import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

LOGGER_NAME = "ImageViewer"


def view_image(path: str, index: int):
    images = np.load(file=path)

    image = images[:, :, index]

    plt.figure()
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(LOGGER_NAME)

    parser = argparse.ArgumentParser(
        description="This is a tool for renaming the open-review images to a different size")
    parser.add_argument("--npy_path",
                        help="path to npy file of images to view",
                        required=True,
                        type=str)
    parser.add_argument("--index",
                        help="Index of image to view",
                        required=True,
                        type=int)

    args = parser.parse_args()
    view_image(args.npy_path, args.index)
