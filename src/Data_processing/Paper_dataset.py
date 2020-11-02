from torch.utils.data import *
from os import path
from src.Tools.open_review_dataset import *

import pandas as pd
import torchvision


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

        # Set global path and path to meta file
        self.data_path = data_path
        path_meta = os.path.join(data_path, "meta.csv")

        self.four_dim = False

        if dataset_type == Mode.RGBFrontPage:
            self.dataset_type = "-rgb-frontpage-"
        elif dataset_type == Mode.GSFrontPage:
            self.dataset_type = "-gs-frontpage-"
        elif dataset_type == Mode.GSChannels:
            self.dataset_type = "-gs-channels-"
            self.four_dim = True
        elif dataset_type == Mode.RGBChannels:
            self.dataset_type = "-rgb-channels-"
            self.four_dim = True
        elif dataset_type == Mode.RGBBigImage:
            self.dataset_type = "-rgb-bigimage-"
        elif dataset_type == Mode.GSBigImage:
            self.dataset_type = "-gs-bigimage-"
        else:
            print("NO VALID DATASET")
            exit()

        self.res = str(width) + "-" + str(height)
        self.csv_data, self.len = self.create_usable_csv(path_meta)

        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.functional.to_tensor,
             torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

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

        #remove = len(remove_element)
        csv = csv.drop(index=remove_element)
        #length2 = len(csv.index)


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

        image = image / 255

        #print(image)
        #print(image.shape)
        #print(type(image))
        image = self.transform(image)
        #print(image)
        #print(image.shape)
        #print(type(image))


        # This line might be needed by pytorch to switch place for the channel data
        #if not self.four_dim:
        #    image = image.transpose((2, 0, 1))
        #else:
        #    image = image.transpose((0, 3, 1, 2))

        ret["image"] = image
        if data["accepted"]:
            ret["label"] = np.asarray([1.0])
        else:
            ret["label"] = np.asarray([0.0])

        ret["abstract"] = data["abstract"]
        ret["title"] = data["title"]
        ret["authors"] = data["authors"]

        return ret
