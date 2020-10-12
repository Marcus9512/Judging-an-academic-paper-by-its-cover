import torch
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as ut
import torch.utils.tensorboard as tb
from datetime import datetime

from src.Data_processing.Paper_dataset import *


class Trainer:

    def __init__(self, dataset, logger, use_gpu=True, data_to_train=0.5, data_to_test=0.25, data_to_eval=0.25):
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
        self.logger = logger
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

    def train(self, model, batch_size, learn_rate, learn_decay, learn_momentum, epochs, image_type):

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

        self.logger.info(f"------Training--------")
        self.logger.info(f"\t Epochs \t{epochs}")
        self.logger.info(f"\t Batch \t\t{batch_size}")
        self.logger.info(f"\t Learn rate \t{learn_rate}")
        self.logger.info(f"\t Momentum \t{learn_momentum}")
        self.logger.info(f"\t Decay \t\t{learn_decay}")

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

        self.logger.info(f"\t Size of Traindata: \t{len(dataloader_train) * batch_size}")
        self.logger.info(f"\t Size of Validata: \t{len(dataloader_val) * batch_size}")
        self.logger.info(f"\t Size of Testdata: \t{len(dataloader_test) * batch_size}")

        summary = tb.SummaryWriter()

        #Train model
        model = self.train_and_evaluate_model(model, dataloader_train, dataloader_val, summary,
                                      epochs, learn_rate, learn_decay, learn_momentum, image_type)
        #Test model
        self.test_model(model, dataloader_test)

        summary.flush()
        summary.close()

    def save_model(self, model, image_type):
        path = "saved_nets"
        if not os.path.exists(path):
            os.mkdir(path)
        t = datetime.now()
        t = t.strftime("%d-%m-%Y-%H-%M-%S")
        path = path + "/"+image_type+"-"+t
        torch.save(model.state_dict(), path)

    def train_and_evaluate_model(self, model, dataloader_train, dataloader_val, summary,
                                 epochs, learn_rate, learn_decay, learn_momentum, image_type):


        # select optimizer type, current is SGD
        optimizer = opt.SGD(model.parameters(), lr=learn_rate, weight_decay=learn_decay, momentum=learn_momentum)

        evaluation = nn.BCEWithLogitsLoss()  # if binary classification use BCEWithLogitsLoss

        for e in range(epochs):

            self.logger.info(f"Epoch: {e} of: {epochs}")

            #Switch between training and validation
            for session in ["training", "validation"]:

                if session == "training":
                    current = dataloader_train
                    model.train()
                else:
                    current = dataloader_val
                    model.eval()

                total_loss = 0
                elements = 0

                for _, data in enumerate(current):
                    batch = data["image"]
                    label = data["label"]
                    batch = batch.to(device=self.main_device, dtype=torch.float32)
                    label = label.to(device=self.main_device, dtype=torch.float32)

                    if session == "training":
                        optimizer.zero_grad()
                        out = model(batch)
                        loss = evaluation(out, label)
                    else:
                        with torch.no_grad():
                            out = model(batch)
                            loss = evaluation(out, label)

                    total_loss += loss.item()

                    if session == "training":
                        loss.backward()
                        optimizer.step()

                    if (elements % 1000 == 0):
                        self.logger.info(f"{session} img: {elements}")

                    elements += label.size(0)

                total_loss /= elements
                self.logger.info(f"{session} loss: {total_loss}")
                summary.add_scalar('Loss/' + session, total_loss, e)

        self.save_model(model, image_type)
        return model

    def test_model(self, model, test_dataloder, print_res=True):
        correct = 0
        total = 0
        len_test = len(test_dataloder)

        self.logger.info(f"------Test--------")
        self.logger.info(f"Test, number of samples: {len_test}")
        model.eval()
        for i in test_dataloder:

            test = i["image"]
            label = i["label"]

            with torch.no_grad():
                test = test.to(device=self.main_device, dtype=torch.float32)
                out = model(test)
                label = label.to(device=self.main_device, dtype=torch.float32)

                out = out.cpu().detach().numpy()
                label = label.cpu().detach().numpy()

                for element in range(len(label)):
                    pred = 1.0 if out[element][0] > 0 else 0.0
                    found = False
                    if label[element][0] == pred:
                        correct += 1
                        found = True
                    total += 1

                    if print_res:
                        self.logger.info(
                            f"Output from network, predicted: {pred}, label: {label[element][0]}, out: {out[element][0]}, "
                            f"correct: {found}, total correct: {correct}, total: {total}")


        self.logger.info(f"Accuracy: {(correct/total)*100}%")

'''
 # Code graveyard
 #TODO Refactor to only have one loop for both validation and training
        for e in range(epochs):
            self.logger.info(f"Epoch: {e} of: {epochs}")
            loss_training = 0

            # Training
            model.train()
            pos = 0
            for i in dataloader_train:
                train = i["image"]
                label = i["label"]

                # reset gradients
                optimizer.zero_grad()
                train = train.to(device=self.main_device, dtype=torch.float32)
                out = model(train)

                label = label.to(device=self.main_device, dtype=torch.float32)

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

                    label_val = label_val.to(device=self.main_device, dtype=torch.float32)

                    loss = evaluation(out, label_val)
                    loss_val += loss.item()

                    pos += 1

            loss_val /= len_v

            self.logger.info(f"Training loss: {loss_training} Validation loss: {loss_val}")

            summary.add_scalar('Loss/train', loss_training, e)
            summary.add_scalar('Loss/val', loss_val, e)
            
            
            
    
'''