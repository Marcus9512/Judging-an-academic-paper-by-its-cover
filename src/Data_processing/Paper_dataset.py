import os
import numpy as np
from torch.utils.data import *
from PIL import Image
import pandas as pd

class Paper_dataset(Dataset):

    '''
        Creates a "Paper_dataset" from the data given in meta.
        At the moment the file
    '''
    def __init__(self, data_path, print_csv=False):
        #Set global path and path to meta file
        self.data_path = data_path
        self.path_meta = os.path.join(data_path,"meta.csv")
        if not os.path.exists(self.path_meta):
            raise Exception("Could not find meta file")

        #Save csv and number of rows
        csv = pd.read_csv(self.path_meta)
        self.len = len(csv.index)
        self.csv_data = csv

        if print_csv:
            print(csv.index)
            print(csv)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        #Load data from row item, note that the entire csv file is located in the memory, might be a problem if
        #the csv file gets to big
        data = self.csv_data.loc[item,:]

        ret = {}
        image = Image.open(self.data_path+"/"+data["image_path"])
        image = np.asarray(image) / 255

        #This line might be needed by pytorch to switch place for the channel data
        #image = image.transpose((2, 0, 1))

        ret["image"] = image
        ret["label"] = data["accepted"]
        ret["abstract"] = data["abstract"]
        ret["title"] = data["title"]
        ret["authors"] = data["authors"]

        return ret
