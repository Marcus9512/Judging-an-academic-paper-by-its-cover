import numpy as np
from PIL import Image
from os import listdir, makedirs
from os.path import isfile, join, exists
import re
from imageio import *
import argparse

def convert_images_to_png(papers_path, regex=".*bigimage.*"):
    only_files = [f for f in listdir(papers_path) if isfile(join(papers_path, f))]
    r = re.compile(regex)
    matched_images = list(filter(r.match, only_files))
    print("################################")
    print(f"Saving the following images from directory {papers_path}: ", matched_images)
    
    img_directory = f"{papers_path}/png_images"

    if not exists(img_directory):
        makedirs(img_directory)

    for img in matched_images:
        img_array = np.load(f"{papers_path}/{img}")
        img_name = img.split(".")[0]
        img_uint8 = img_array.astype(np.uint8)
        imsave(f"{img_directory}/{img_name}.png", img_uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_convert_path", type=str, help="The path to the npy images you wish to convert to png")
    parser.add_argument('--regex', nargs='?', default=".*bigimage.*")
    args = parser.parse_args()

    convert_images_to_png(args.img_convert_path, args.regex)