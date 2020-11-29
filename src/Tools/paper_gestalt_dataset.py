import pandas as pd
import argparse
from PIL import Image
import pathlib
import logging
import numpy as np
import os
from tqdm import tqdm

LOGGER_NAME = "GESTALT"


class RemoveFromPage:
    @staticmethod
    def remove_page(paper: np.ndarray, n: int) -> np.ndarray:
        paper = np.array(paper)

        x_page_offset = 165
        x_page_size = 185
        x_page_number = n % 4

        y_page_number = int(n / 4)
        y_page_offset = 220
        y_page_size = 200

        paper[y_page_number * y_page_offset:y_page_number * y_page_offset +
              y_page_size, x_page_number *
              x_page_offset:x_page_number * x_page_offset + x_page_size] = 0

        return paper

    @staticmethod
    def numbers(paper: np.ndarray) -> np.ndarray:
        paper = np.array(paper)
        paper[193:223, 70:100] = 0
        paper[193:223, 240:270] = 0
        paper[193:223, 410:440] = 0
        paper[193:223, 580:610] = 0

        paper[413:443, 70:100] = 0
        paper[413:443, 240:270] = 0
        paper[413:443, 410:440] = 0
        paper[413:443, 580:610] = 0
        return paper

    @staticmethod
    def remove_box(paper: np.ndarray, x, y):
        paper = np.array(paper)
        paper[y:y + 30, x:x + 30] = 0
        return paper

def convert_gestalt_to_rgb_bigimage(
        gestalt_root: pathlib.Path, data_root: pathlib.Path,
        remove_side_number: bool, remove_front_page: bool,
        remove_last_two_pages: bool, boxes_to_remove: int,
        pages_to_remove: int):
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

    x, y = None, None
    if boxes_to_remove:
        x = np.random.randint(0, 680, size=boxes_to_remove)
        y = np.random.randint(0, 440, size=boxes_to_remove)

    n = None
    if pages_to_remove:
        n = np.random.randint(0, 7, pages_to_remove)

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

            if remove_last_two_pages:
                im = np.array(im)
                im = RemoveFromPage.remove_page(im, 6)
                im = RemoveFromPage.remove_page(im, 7)
                im = Image.fromarray(im)

            if remove_front_page:
                im = np.array(im)
                im = RemoveFromPage.remove_page(im, 0)
                im = Image.fromarray(im)

            if remove_side_number:
                im = np.array(im)
                im = RemoveFromPage.numbers(im)
                im = Image.fromarray(im)

            if pages_to_remove:
                im = np.array(im)
                for i in range(pages_to_remove):
                    im = RemoveFromPage.remove_page(im, n[i])
                im = Image.fromarray(im)

            if boxes_to_remove:
                im = np.array(im)
                for i in range(boxes_to_remove):
                    im = RemoveFromPage.remove_box(im, x[i], y[i])
                im = Image.fromarray(im)

            im = im.resize((256 * 4, 256 * 2))
            arr = np.array(im)

            np.save(
                str(binary_blob_output /
                    f"{im_name}-rgb-bigimage-256-256.npy"), arr)

            paths.append(relative_path + im_name)
            accepted.append(label)

            if is_test:
                year.append("2020")
            else:
                year.append("2018")

    pd.DataFrame({
        "paper_path": paths,
        "accepted": accepted,
        "year": year
    }).to_csv(str(data_root / "meta.csv"), index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(LOGGER_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gestalt_root", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--remove_side_number", action="store_true")
    parser.add_argument("--remove_front_page", action="store_true")
    parser.add_argument("--remove_last_two_pages", action="store_true")
    parser.add_argument("--random_boxes_to_remove", type=int)
    parser.add_argument("--random_pages_to_remove", type=int)

    args = parser.parse_args()

    convert_gestalt_to_rgb_bigimage(
        pathlib.Path(args.gestalt_root),
        pathlib.Path(args.data_root),
        remove_side_number=args.remove_side_number,
        remove_front_page=args.remove_front_page,
        remove_last_two_pages=args.remove_last_two_pages,
        boxes_to_remove=args.random_boxes_to_remove,
        pages_to_remove=args.random_pages_to_remove)
