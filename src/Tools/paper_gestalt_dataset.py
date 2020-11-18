import pandas as pd
import argparse
from PIL import Image
import pathlib
import logging
import numpy as np
import os
from tqdm import tqdm

LOGGER_NAME = "GESTALT"


def convert_gestalt_to_rgb_bigimage(gestalt_root: pathlib.Path,
                                    data_root: pathlib.Path,
                                    remove_side_number: bool):
    logger = logging.getLogger(LOGGER_NAME)

    if not gestalt_root.is_dir():
        logger.fatal(f"{gestalt_root} is not a valid directory")

    if not data_root.is_dir():
        logger.fatal(f"{data_root} is not a valid directory")

    train_accepted = gestalt_root / 'train' / 'conference'
    train_rejected = gestalt_root / 'train' / 'workshop'

    test_accepted = gestalt_root / 'test' / 'conference'
    test_rejected = gestalt_root / 'test' / 'workshop'

    if not train_accepted.is_dir():
        logger.fatal(f"{train_accepted} is not a valid directory")

    if not train_rejected.is_dir():
        logger.fatal(f"{train_rejected} is not a valid directory")

    if not test_accepted.is_dir():
        logger.fatal(f"{test_accepted} is not a valid directory")

    if not test_rejected.is_dir():
        logger.fatal(f"{test_rejected} is not a valid directory")

    binary_blob_output = data_root / 'papers'

    if not binary_blob_output.is_dir():
        os.mkdir(binary_blob_output)

    relative_path = 'papers/'

    accepted = []
    paths = []
    year = []
    for is_test, path, label in [(False, train_accepted, True),
                                 (True, test_accepted, True),
                                 (False, train_rejected, False),
                                 (True, test_rejected, False)]:
        for paper in tqdm(list(path.glob("*"))):
            im_name = paper.stem
            im = Image.open(str(paper), mode='r')
            if remove_side_number:
                im = np.array(im)

                im[203:213, 80:90] = 0
                im[203:213, 250:260] = 0
                im[203:213, 420:430] = 0
                im[203:213, 590:600] = 0

                im[423:433, 80:90] = 0
                im[423:433, 250:260] = 0
                im[423:433, 420:430] = 0
                im[423:433, 590:600] = 0

                im = Image.fromarray(im)

            im = im.resize((256 * 4, 256 * 2))
            arr = np.array(im)

            np.save(str(binary_blob_output / f"{im_name}-rgb-bigimage-256-256.npy"), arr)

            paths.append(relative_path + im_name)
            accepted.append(label)

            if is_test:
                year.append("2020")
            else:
                year.append("2018")

    pd.DataFrame({"paper_path": paths, "accepted": accepted, "year": year}
                 ).to_csv(str(data_root / "meta.csv"), index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(LOGGER_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gestalt_root", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--remove_side_number", action="store_true")

    args = parser.parse_args()

    convert_gestalt_to_rgb_bigimage(pathlib.Path(args.gestalt_root), pathlib.Path(args.data_root), remove_side_number=args.remove_side_number)
