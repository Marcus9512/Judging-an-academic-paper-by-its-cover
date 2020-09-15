'''
Driver class for the network.
Our "main" method.
'''

import logging
import argparse
from src.Data_processing.Paper_dataset import *
from src.Tools.Tools import *


LOGGER_NAME = "Network_Driver"

def sanity_check_paper_dataset(dataset_path):
    '''
    Sanity check for the Paper_dataset class
    :param dataset_path:
    :return:
    '''
    dataset = Paper_dataset(dataset_path, print_csv=True)

    for i in range(0, dataset.len):
        info = dataset.__getitem__(i)
        print(info["title"])

    info = dataset.__getitem__(2)
    print_image_from_array(info["image"] * 255)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(LOGGER_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, help="Base path of dataset", required=True)
    args = parser.parse_args()
    logger.info(f"Dataset path: {args.base_path} ")

    sanity_check_paper_dataset(args.base_path)