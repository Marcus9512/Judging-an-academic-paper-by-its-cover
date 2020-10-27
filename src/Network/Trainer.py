from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as ut
import torch.utils.tensorboard as tb

from sklearn.metrics import accuracy_score, recall_score, precision_score

from datetime import datetime
from src.Data_processing.Paper_dataset import *

from matplotlib import pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform


EXPERIMENT_LAUNCH_TIME = datetime.now()
# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="rZZFwjbEXYeYOP5J0x9VTUMuf",
                        project_name="dd2430", workspace="dd2430")

class Trainer:

    def __init__(self, dataset, logger, name, use_gpu=True, data_to_train=0.5, data_to_test=0.25, data_to_eval=0.25):
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

        experiment.set_name(name)

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

        #if not model.parameters().is_cuda:
        #    print("Marcus big dumbdumb")

        # Create CAMs
        self.create_CAMs(model, dataloader_val, num_images=1)

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
                # Log to comet
                # experiment.log_metric(f"train {session} - Loss", total_loss)


        self.save_model(model, image_type)
        return model

    def imshow(self, img):
        import matplotlib.pyplot as plt
        #img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # CAM implementation stuff:
    class save_features():
        features = None
        def __init__(self, m): 
            self.hook = m.register_forward_hook(self.hook_fn)

        def hook_fn(self, module, input, output):
            self.features = ((output.cpu()).data).numpy()

        def remove(self):
            self.hook.remove()

    def get_CAM(self, feature_convolution, weights, class_index):
        _, nc, h, w = feature_convolution.shape
        cam = weights[class_index].dot(feature_convolution.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return [cam_img]

    def create_CAM(self, model, image):
        model.eval()
        # model.to(self.main_device)            # <- doesn't work, same error: "Expected device type cuda but got device type cpu"
        # model.gpu()                           # <- resnet has no attribute gpu
        # Det är någonting med image.gpu() /image.to(self.main_device) som ska ske, wonky.

        # setup hook to get last convolutional layer
        final_layer = model._modules.get('layer4')
        activated_features = self.save_features(final_layer)

        # make prediction
        image_pred = image.reshape(1, 3, 512, 1024)
        # type CPU vs GPU?
        prediction = model(image_pred)
        pred_probabilities = F.softmax(prediction).data.squeeze()
        activated_features.remove()

        # get parameters to create CAM
        weight_softmax_params = list(model._modules.get('fc').parameters())
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

        # get prediction from network (probably not necessary with our implementation - should work without this)
        class_idx = topk(pred_probabilities,1)[1].int()

        # create heatmap overlay
        heatmap = self.get_CAM(activated_features.features, weight_softmax, class_idx)

        # show image + overlay
        self.imshow(image)
        plt.imshow(skimage.transform.resize(heatmap[0], tensor.shape[1:3]), alpha=0.5, cmap='jet');
        
        path = "saved_CAMs"
        if not os.path.exists(path):
            os.mkdir(path)
        t = datetime.now()
        t = t.strftime("%d-%m-%Y-%H-%M-%S")
        path = path + "/"+image_type+"-"+t+'.png'

        plt.savefig(path)
        plt.close()


    # TODO: set this to random
    def create_CAMs(self, model, dataloader_val, num_images=10):
        for i in range(num_images):
            # next(iter(dataloader_val)) returns a dict, 'image' is key for the images in the batch.
            image = next(iter(dataloader_val))['image'][0].to(device=self.main_device, dtype=torch.float32)
            # CURRENT TEST ^ .to(device=self.main_device, dtype=torch.float32)
            # print(image.shape):
            # [3, 512, 1024]
            # self.imshow(image)
            # ^works!

            self.create_CAM(model, image)

    def test_from_file(self, model_path, model, test_dataloder):
        model.load_state_dict(torch.load(model_path))
        self.test_model(model=model, test_dataloder=test_dataloder)

    def test_model(self, model, test_dataloder, print_res=True):

        correct = 0
        total = 0
        len_test = len(test_dataloder)

        self.logger.info(f"------Test--------")
        self.logger.info(f"Test, number of samples: {len_test}")
        model.eval()

        preds = np.array([])
        true_pos = np.array([])

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
        accuracy=(correct/total)*100
        self.logger.info(f"Accuracy: {accuracy}%")
        experiment.log_metric(f"test - accuracy", accuracy)

        self.logger.info(f"True_pos {true_pos}")
        self.logger.info(f"Preds {preds}")

        recall = recall_score(y_true=true_pos, y_pred=preds)
        precision = precision_score(y_true=true_pos, y_pred=preds)

        self.logger.info(f"test -- recall: {recall} -- precision: {precision} ")

        # Log to comet
        experiment.log_metric(f"test - recall", recall)
        experiment.log_metric(f"test - precision", precision)

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