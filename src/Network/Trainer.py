from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as ut

from sklearn.metrics import accuracy_score, recall_score, precision_score

from datetime import datetime
from src.Data_processing.Paper_dataset import *
from tqdm import tqdm

EXPERIMENT_LAUNCH_TIME = datetime.now()


class Trainer:

    def __init__(self, dataset, logger, dataset_type, network_type, use_gpu=True, data_to_train=0.7, data_to_test=0.1,
                 data_to_eval=0.2, log_to_comet=True):
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

        self.log_to_comet = log_to_comet

        if log_to_comet:
            self.experiment = Experiment(api_key="rZZFwjbEXYeYOP5J0x9VTUMuf",
                                         project_name="dd2430", workspace="dd2430")

            self.experiment.set_name(network_type.value)
            self.experiment.add_tag(dataset_type.value)
            self.experiment.add_tag(network_type.value)

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

    def train(self, model, batch_size, learn_rate, epochs, image_type):

        '''
        Performs a train and test cycle at the given model
        :param model: a trainable pytorch model
        :param batch_size: wanted size of each batch
        :param learn_rate: wanted learning rate
        :param epochs: number of epochs
        :return:
        '''

        self.logger.info(f"------Training--------")
        self.logger.info(f"\t Epochs \t{epochs}")
        self.logger.info(f"\t Batch \t\t{batch_size}")
        self.logger.info(f"\t Learn rate \t{learn_rate}")

        # assign model to main device
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

        dataloader_train = ut.DataLoader(batch_train,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         pin_memory=True)
        dataloader_val = ut.DataLoader(batch_val,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       pin_memory=True)
        dataloader_test = ut.DataLoader(batch_test,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True)

        self.logger.info(f"\t Size of Traindata: \t{len(dataloader_train) * batch_size}")
        self.logger.info(f"\t Size of Validata: \t{len(dataloader_val) * batch_size}")
        self.logger.info(f"\t Size of Testdata: \t{len(dataloader_test) * batch_size}")

        # Train model
        model = self.train_and_evaluate_model(model,
                                              dataloader_train,
                                              dataloader_val,
                                              epochs,
                                              learn_rate,
                                              image_type=image_type)
        # Test model
        self.test_model(model, dataloader_train, prefix="train")
        self.test_model(model, dataloader_val, prefix="validation")
        self.test_model(model, dataloader_test, prefix="test")

    def save_model(self, model, image_type):
        path = "saved_nets"
        if not os.path.exists(path):
            os.mkdir(path)
        t = datetime.now()
        t = t.strftime("%d-%m-%Y-%H-%M-%S")
        path = path + "/" + image_type + "-" + t
        torch.save(model.state_dict(), path)

    def train_and_evaluate_model(self,
                                 model,
                                 dataloader_train,
                                 dataloader_val,
                                 epochs,
                                 learn_rate,
                                 image_type,
                                 eval_every: int = 5):

        # select optimizer type, current is SGD
        optimizer = opt.Adam(model.parameters(), lr=learn_rate)

        evaluation = nn.BCEWithLogitsLoss()  # if binary classification use BCEWithLogitsLoss

        i_batch = 0
        train_loss = 0
        with tqdm(desc="Epochs", total=epochs) as epoch_progress_bar:
            for e in range(epochs):
                # initialize as train
                model.train()
                with tqdm(desc="Train", total=len(dataloader_train)) as train_progress_bar:
                    for _, data in enumerate(dataloader_train):
                        i_batch += 1

                        batch = data["image"].to(device=self.main_device, dtype=torch.float32)
                        label = data["label"].to(device=self.main_device, dtype=torch.float32)

                        # 'eval_every' batch we evaluate
                        if i_batch % eval_every == 0:
                            with tqdm(desc="Evaluation", total=len(dataloader_val)) as eval_progress_bar:
                                # Set eval mode
                                model.eval()

                                eval_loss = 0
                                for eval_data in dataloader_val:
                                    eval_batch = eval_data["image"].to(device=self.main_device, dtype=torch.float32)
                                    eval_label = eval_data["label"].to(device=self.main_device, dtype=torch.float32)

                                    with torch.no_grad():
                                        out = model(eval_batch)
                                        loss = evaluation(out, eval_label)

                                    eval_progress_bar.update()
                                    eval_loss += loss.item()

                                eval_progress_bar.write(f"eval - Loss: {eval_loss / eval_every}")
                                eval_progress_bar.write(f"train - Loss: {train_loss / eval_every}")

                                # Log to comet
                                if self.log_to_comet:
                                    self.experiment.log_metric(f"eval - Loss", eval_loss / eval_every)
                                    self.experiment.log_metric(f"train - Loss", train_loss / eval_every)

                                # Set back to train
                                model.train()

                                # Reset train_loss
                                train_loss = 0

                        optimizer.zero_grad()
                        out = model(batch)
                        loss = evaluation(out, label)

                        train_loss += loss.item()

                        loss.backward()
                        optimizer.step()

                        train_progress_bar.update()

                epoch_progress_bar.update()
            self.save_model(model, image_type)
        return model

    def test_from_file(self, model_path, model, dataloder, prefix: str):
        model.load_state_dict(torch.load(model_path))
        self.test_model(model=model, dataloader=dataloder, prefix=prefix)

    def test_model(self, model, dataloader, prefix: str, print_res=False):

        correct = 0
        total = 0
        len_test = len(dataloader)

        self.logger.info(f"------{prefix}--------")
        self.logger.info(f"{prefix}, number of samples: {len_test}")
        model.eval()

        preds = np.array([])
        true_pos = np.array([])

        for i in dataloader:

            test = i["image"]
            label = i["label"]

            with torch.no_grad():
                test = test.to(device=self.main_device, dtype=torch.float32)
                out = model(test)
                label = label.to(device=self.main_device, dtype=torch.float32)

                out = out.cpu().detach().numpy()
                label = label.cpu().detach().numpy()

                for element in range(len(label)):
                    pred = 1.0 if out[element][0] > 0.5 else 0.0
                    found = False
                    preds = np.append(preds, pred)
                    true_pos = np.append(true_pos, label[element][0])
                    if label[element][0] == pred:
                        correct += 1
                        found = True
                    total += 1

                    if print_res:
                        self.logger.info(
                            f"Output from network, predicted: {pred}, label: {label[element][0]}, out: {out[element][0]}, "
                            f"correct: {found}, total correct: {correct}, total: {total}")

        accuracy = accuracy_score(y_true=true_pos, y_pred=preds)
        recall = recall_score(y_true=true_pos, y_pred=preds)
        precision = precision_score(y_true=true_pos, y_pred=preds)

        self.logger.info(f"Accuracy: {accuracy}%")
        self.logger.info(f"Recall: {recall}")
        self.logger.info(f"Precision: {precision}%")

        # Log to comet
        if self.log_to_comet:
            self.experiment.log_metric(f"{prefix} - accuracy", accuracy)
            self.experiment.log_metric(f"{prefix} - recall", recall)
            self.experiment.log_metric(f"{prefix} - precision", precision)
