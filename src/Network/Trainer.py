import torch
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as ut
import torch.utils.tensorboard as tb

from src.Data_processing.Paper_dataset import *


class Trainer:

    def __init__(self, dataset, use_gpu=True, data_to_train=0.5, data_to_test=0.25, data_to_eval=0.25):
        '''
        :param data_path: path to the data folder
        :param use_gpu: true if the program should use GPU
        :param data_to_train: percent of data to train
        :param data_to_test: percent of data to test
        :param data_to_eval: percent of data to eval
        '''
        self.dataset = dataset
        self.data_to_train = data_to_train
        self.data_to_test = data_to_test
        self.data_to_eval = data_to_eval
        self.main_device = self.get_main_device(use_gpu)

    def get_main_device(self, use_gpu):
        '''
           Initilize class by loading data and maybe preprocess
           ASSIGN CUDAS IF POSSIBLE
           :return:
        '''

        # Check if gpu is available
        if torch.cuda.is_available() and use_gpu:
            device = "cuda:0"
            print("Using GPU")
            self.check_gpu_card()
        else:
            device = "cpu"
            print("Using CPU")

        # assign gpu or cpu to the main_device
        main_device = torch.device(device)

        return main_device

    def check_gpu_card(self):
        '''
        The method tries to check which gpu you are using on your computer
        :return:
        '''
        try:
            import pycuda.driver as cudas
            print("The device you are using is: ", cudas.Device(0).name())

        except ImportError as e:
            print("Could not find pycuda and thus not show amazing stats about youre GPU, have you installed CUDA?")
            pass

    def train(self, model, batch_size, learn_rate, learn_decay, learn_momentum, epochs):

        '''
        Performs a train and test cycle at the given model
        :param model: a trainable pytorch model
        :param batch_size: wanted size of each batch
        :param learn_rate: wanted learning rate
        :param learn_decay: wanted learn decay
        :param learn_momentum: wanted learn momentum
        :param epochs: number of epochs
        :return:
        '''

        #assign model to main device
        model.to(self.main_device)

        batch_train = self.dataset
        dataset_length = batch_train.len

        # convert to int
        to_train = int(dataset_length * self.data_to_train)
        to_test = int(dataset_length * self.data_to_test)
        to_val = int(dataset_length * self.data_to_eval)

        # make sure that the sum of data is not more than available data
        data_sum = to_train + to_test + to_val
        if data_sum < dataset_length:
            to_val += dataset_length - data_sum

        # split data to train, validation and test.
        batch_train, batch_val, batch_test = random_split(batch_train, [to_train, to_val, to_test])

        dataloader_train = ut.DataLoader(batch_train, batch_size=batch_size, shuffle=True, pin_memory=True)
        dataloader_val = ut.DataLoader(batch_val, batch_size=batch_size, shuffle=True, pin_memory=True)
        dataloader_test = ut.DataLoader(batch_test, batch_size=batch_size, shuffle=True, pin_memory=True)

        len_t = len(dataloader_train)
        len_v = len(dataloader_val)
        len_test = len(dataloader_test)

        # select optimizer type, current is SGD
        optimizer = opt.SGD(model.parameters(), lr=learn_rate, weight_decay=learn_decay, momentum=learn_momentum)

        evaluation = nn.CrossEntropyLoss()

        summary = tb.SummaryWriter()

        # Training loop

        for e in range(epochs):
            print("Epoch: ", e, " of ", epochs)
            loss_training = 0

            # Training
            model.train()
            pos = 0
            for i in dataloader_train:
                train = i["image"]
                label = i["label"]

                #print(train.shape)
                #print("Train ",train.type())
                # reset gradients
                optimizer.zero_grad()
                train = train.to(device=self.main_device, dtype=torch.float32)
                #print("Train2 ", train.type())
                out = model(train)

                label = label.to(device=self.main_device, dtype=torch.long)

                #print("out ", out.type())
                #print(out)
                #print("label", label.type())
                #print(label)
                loss = evaluation(out, label)
                loss.backward()
                optimizer.step()

                loss_training += loss.item()
                pos += 1

            loss_training /= len_t
            loss_val = 0

            # Validation
            model.eval()
            pos = 0
            for j in dataloader_val:
                val = j["image"]
                label_val = j["label"]
                val = val.to(device=self.main_device, dtype=torch.float32)

                with torch.no_grad():
                    out = model(val)

                    label_val = label_val.to(device=self.main_device, dtype=torch.long)

                    loss = evaluation(out, label_val)
                    loss_val += loss.item()
                    pos += 1

            loss_val /= len_v

            print("Training loss: ", loss_training)
            print("Validation loss: ", loss_val)

            summary.add_scalar('Loss/train', loss_training, e)
            summary.add_scalar('Loss/val', loss_val, e)

        summary.flush()
        summary.close()

        # Evaluation
