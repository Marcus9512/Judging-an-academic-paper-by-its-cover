from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as ut

from sklearn.metrics import accuracy_score, recall_score, precision_score

from datetime import datetime
from src.Data_processing.Paper_dataset import *
from tqdm import tqdm

from matplotlib import pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform

EXPERIMENT_LAUNCH_TIME = datetime.now()


class Trainer:

    def __init__(self, train_dataset, test_dataset, logger, dataset_type, network_type, pretrained, freeze, use_gpu=True,
                 data_to_train=0.7,
                 data_to_eval=0.3,
                 log_to_comet=True,
                 create_heatmaps=False):
        '''
        :param data_path: path to the data folder
        :param use_gpu: true if the program should use GPU
        :param data_to_train: percent of data to train
        :param data_to_eval: percent of data to eval
        '''
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.data_to_train = data_to_train
        self.data_to_eval = data_to_eval
        self.logger = logger
        self.main_device = self.get_main_device(use_gpu)
        self.create_heatmaps = create_heatmaps
        self.log_to_comet = log_to_comet

        self.freeze = freeze

        if log_to_comet:
            self.experiment = Experiment(api_key="rZZFwjbEXYeYOP5J0x9VTUMuf",
                                         project_name="dd2430", workspace="dd2430",
                                         auto_metric_logging=False,
                                         auto_param_logging=False,
                                         auto_output_logging=False)

            self.experiment.set_name(network_type.value)
            self.experiment.add_tag(dataset_type.value)
            self.experiment.add_tag(network_type.value)

            if pretrained:
                self.experiment.add_tag("pre-trained")
            if freeze:
                self.experiment.add_tag("freeze")

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

        # convert to int
        to_train = int(self.train_dataset.len * self.data_to_train)
        to_val = int(self.train_dataset.len * self.data_to_eval)

        # make sure that the sum of data is not more than available data
        data_sum = to_train + to_val
        if data_sum < self.train_dataset.len:
            to_val += self.train_dataset.len - data_sum

        # split data to train, validation and test.
        batch_train, batch_val = random_split(self.train_dataset, [to_train, to_val])

        dataloader_train = ut.DataLoader(batch_train,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         pin_memory=True)
        dataloader_val = ut.DataLoader(batch_val,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       pin_memory=True)
        dataloader_test = ut.DataLoader(self.test_dataset,
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

        # Custom dataloader to create CAMs - batch size set to 1
        dataloader_val_cam = ut.DataLoader(batch_val,
                                           batch_size=1,
                                           shuffle=True,
                                           pin_memory=True)

        if self.create_heatmaps:
            self.create_CAMs(model, dataloader_val_cam, image_type)

        # Test model
        self.test_model(model, dataloader_train, batch_size=batch_size, prefix="train")
        self.test_model(model, dataloader_val, batch_size=batch_size, prefix="validation")
        self.test_model(model, dataloader_test, batch_size=batch_size, prefix="test")

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
                                 eval_every: int = 50):

        # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        params_to_update = model.parameters()
        if self.freeze:
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)

        # select optimizer type, current is SGD
        optimizer = opt.Adam(params_to_update, lr=learn_rate, weight_decay=1e-6)

        evaluation = nn.BCEWithLogitsLoss()  # if binary classification use BCEWithLogitsLoss

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(dataloader_train), 0.000001)

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

                                eval_progress_bar.write(f"eval - Loss: {eval_loss / len(dataloader_val)}")
                                eval_progress_bar.write(f"train - Loss: {train_loss / eval_every}")

                                # Log to comet
                                if self.log_to_comet:
                                    self.experiment.log_metric(f"eval - Loss", eval_loss / len(dataloader_val))
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
                        scheduler.step()

                        train_progress_bar.update()

                epoch_progress_bar.update()
            self.save_model(model, image_type)
        return model

    # CAM implementation stuff:
    class save_features():
        features = None

        def __init__(self, m):
            self.hook = m.register_forward_hook(self.hook_fn)

        def hook_fn(self, module, input, output):
            self.features = ((output.cpu()).data).numpy()

        def remove(self):
            self.hook.remove()

    def get_CAM(self, feature_convolution, weights):
        _, nc, h, w = feature_convolution.shape
        cam = weights.dot(feature_convolution.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return [cam_img]

    def create_CAM(self, model, image, image_type, label):
        model.eval()

        # setup hook to get last convolutional layer
        final_layer = model._modules.get('layer4')
        activated_features = self.save_features(final_layer)

        # make prediction
        image_pred = image.reshape(1, 3, 512, 1024)
        prediction = model(image_pred)
        pred_probabilities = F.softmax(prediction).data.squeeze()
        activated_features.remove()

        # get parameters to create CAM
        weight_softmax_params = list(model._modules.get('fc').parameters())
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

        # create heatmap overlay
        heatmap = self.get_CAM(activated_features.features, weight_softmax)

        # show image + overlay
        plt.imshow(np.transpose(image.cpu().data.numpy(), (1, 2, 0)))

        prediction = prediction.cpu().detach().numpy()[0]
        prediction = 1.0 if prediction > 0.5 else 0.0
        if prediction == 0.0:
            plt.title(image_type + ', prediction=0.0' + ', label=' + str(label))
            plt.imshow(skimage.transform.resize(heatmap[0], image.shape[1:3]), alpha=0.5, cmap='jet_r')
        else:
            plt.title(image_type + ', prediction=1.0' + ', label=' + str(label))
            plt.imshow(skimage.transform.resize(heatmap[0], image.shape[1:3]), alpha=0.5, cmap='jet')

        if self.log_to_comet:
            self.experiment.log_figure(figure_name=image_type + ', prediction=' + str(prediction) + ', label=' + str(label))

        path = "saved_CAMs"
        if not os.path.exists(path):
            os.mkdir(path)
        t = datetime.now()
        t = t.strftime("%d-%m-%Y-%H-%M-%S")
        path = path + "/" + image_type + "-" + t + '.png'

        plt.savefig(path)
        plt.close()

    def create_CAMs(self, model, dataloader_val, image_type, num_images=2):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        while true_negative < num_images or true_positive < num_images or false_positive < num_images or false_negative < num_images:
            # next(iter(dataloader_val)) returns a dict, 'image' is key for the images in the batch.
            data = next(iter(dataloader_val), None)

            if data == None:
                logger.info(f"CAM could not create 2 images of all types - returning to run tests")
                return

            image = data['image'][0].to(device=self.main_device, dtype=torch.float32)
            
            label = data['label'][0].to(device=self.main_device, dtype=torch.float32)
            label = label.cpu().detach().numpy()[0]

            image_pred = image.reshape(1, 3, 512, 1024)
            prediction = model(image_pred)
            prediction = prediction.cpu().detach().numpy()[0]
            prediction = 1.0 if prediction > 0.5 else 0.0

            if label == 0.0 and prediction == 0.0 and true_negative < num_images:
                self.create_CAM(model, image, image_type, label)
                true_negative = true_negative + 1

            if label == 1.0 and prediction == 0.0 and false_negative < num_images:
                self.create_CAM(model, image, image_type, label)
                false_negative = false_negative + 1

            if label == 1.0 and prediction == 1.0 and true_positive < num_images:
                self.create_CAM(model, image, image_type, label)
                true_positive = true_positive + 1

            if label == 0.0 and prediction == 1.0 and false_positive < num_images:
                self.create_CAM(model, image, image_type, label)
                false_positive = false_positive + 1


    def test_from_file(self, model_path, model, dataloader, prefix: str):
        model.load_state_dict(torch.load(model_path))
        self.test_model(model=model, dataloader=dataloader, prefix=prefix)

    def test_model(self, model, dataloader, batch_size: int, prefix: str, print_res=False):

        correct = 0
        total = 0
        pred_acc = 0
        pred_rej = 0

        len_test = len(dataloader)

        self.logger.info(f"------{prefix}--------")
        self.logger.info(f"{prefix}, number of samples: {len_test * batch_size}")
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
                #print(out.shape)
                out = out.cpu().detach().numpy()
                label = label.cpu().detach().numpy()

                for element in range(len(label)):
                    pred = 1.0 if out[element][0] > 0.5 else 0.0
                    found = False
                    preds = np.append(preds, pred)
                    true_pos = np.append(true_pos, label[element][0])

                    if pred == 1:
                        pred_acc+=1
                    else:
                        pred_rej+=1

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

        self.logger.info(f"Predicted accept {pred_acc / total}")
        self.logger.info(f"Predicted reject {pred_rej / total}")

        # Log to comet
        if self.log_to_comet:
            self.experiment.log_metric(f"{prefix} - accuracy", accuracy)
            self.experiment.log_metric(f"{prefix} - recall", recall)
            self.experiment.log_metric(f"{prefix} - precision", precision)

            self.experiment.log_metric(f"{prefix} - Predicted accept", pred_acc / total)
            self.experiment.log_metric(f"{prefix} - Predicted reject", pred_rej / total)
