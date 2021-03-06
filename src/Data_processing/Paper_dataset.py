from torch.utils.data import *
from os import path
from src.Tools.open_review_dataset import *
from randaugment import RandAugment
from src.Data_processing.Augmentations import *

import pandas as pd
import matplotlib.pyplot as plt
from src.Data_processing.Transformation_wraper import *


class Paper_dataset(Dataset):
    '''
        Creates a "Paper_dataset" from the data given in meta.
        At the moment the file
    '''

    def __init__(self, data_path,
                 dataset_type,
                 width,
                 height,
                 print_csv=False,
                 train=True):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(LOGGER_NAME)

        self.logger = logger
        self.train = train

        self.width = width
        self.height = height

        self.transform = Augmentations([0.36, 0, 0.16, 0.16, 0.16, 0.16], self.height, self.width, num_trans=6)

        # Set global path and path to meta file
        self.data_path = data_path
        path_meta = os.path.join(data_path, "meta.csv")

        self.channels = False

        if dataset_type == Mode.RGBFrontPage:
            self.dataset_type = "-rgb-frontpage-"
        elif dataset_type == Mode.GSFrontPage:
            self.dataset_type = "-gs-frontpage-"
        elif dataset_type == Mode.GSChannels:
            self.dataset_type = "-gs-channels-"
            self.channels = True
        elif dataset_type == Mode.RGBChannels:
            self.dataset_type = "-rgb-channels-"
            self.channels = True
        elif dataset_type == Mode.RGBBigImage:
            self.dataset_type = "-rgb-bigimage-"
        elif dataset_type == Mode.GSBigImage:
            self.dataset_type = "-gs-bigimage-"
        else:
            print("NO VALID DATASET")
            exit()

        self.res = str(width) + "-" + str(height)
        self.csv_data, self.len = self.create_usable_csv(path_meta)

        # RandAugment source https://pypi.org/project/randaugment/

        # self.transformations = Transformation_wraper(strong_augmentation, num_transformations)

        if print_csv:
            print(self.csv_data.index)
            print(self.csv_data)

    def get_csv_and_length(self, meta_path):
        '''
        Returns content of a csv and its length
        :return:
        '''
        if not os.path.exists(meta_path):
            raise Exception("Could not find meta file")

            # Save csv and number of rows
        csv = pd.read_csv(meta_path)
        return csv, len(csv.index)

    def create_usable_csv(self, meta_path):
        '''
        Remove files with that does not exist
        :param meta_path:
        :return:
        '''
        csv, length = self.get_csv_and_length(meta_path)

        remove_element = []
        for i in range(length):
            data = csv.loc[i, :]
            p = self.data_path + "/" + data["paper_path"] + self.dataset_type + self.res + ".npy"
            if not path.exists(p):
                remove_element.append(i)

        csv = csv.drop(index=remove_element)

        if self.train:
            train_condition = csv.year != 2020

            csv = csv.loc[train_condition]
        else:
            test_condition = csv.year == 2020

            csv = csv.loc[test_condition]

        self.logger.info(f"Length of CSV {len(csv)}")

        csv = csv.reset_index()
        return csv, len(csv.index)

    def __len__(self):
        return self.len

    def __getitem__(self, item):

        # Load data from row item, note that the entire csv file is located in the memory, might be a problem if
        # the csv file gets to big
        data = self.csv_data.loc[item, :]

        ret = {}

        image = np.load(self.data_path + "/" + data["paper_path"] + self.dataset_type + self.res + ".npy")

        if not self.channels:
            image = Image.fromarray(np.uint8(image))
            if self.train:
                image = self.transform.get_transform()(image)
            else:
                image = self.transform.get_normalisation()(image)
        else:
            image = image.transpose((0,3, 1, 2))
            image = image.reshape((-1, 256, 256))

        ret["image"] = image

        ret['label'] = np.array([data['accepted']], dtype=np.float32)
        return ret
