'''
Driver class for the network.
Our "main" method.
'''

import sys
import os

# Fixes python path for some 
sys.path.append(os.getcwd())

from src.Network.Trainer import *
from src.Tools.open_review_dataset import *

LOGGER_NAME = "Network_Driver"


class Network_type(Enum):
    Resnet = 'resnet18'
    Resnet34 = 'resnet34'

    def __str__(self):
        return self.value


def get_resnet_model(number_of_channels, model, pretrain=True, freeze_pretrain = True):
    # pick resnet 34 and pretrained
    # old resnet 18
    # Copy weights to
    if model != Network_type.Resnet and model != Network_type.Resnet34:
        logger.error(f"Error in get resnet, model is {model}")
        exit()

    if pretrain:
        logger.info(f"Using pretrain")
        model = torch.hub.load('pytorch/vision:v0.6.0', model.value, pretrained=True)

        if freeze_pretrain:
            logger.info(f"Freezing pretrain")
            for param in model.parameters():
                param.requires_grad = False
        #model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        model = torch.hub.load('pytorch/vision:v0.6.0', model.value, pretrained=False, num_classes=1)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 1)
    )
    # modifying the input layer to accept 8 channels input:
    model.conv1 = nn.Conv2d(number_of_channels, 64, kernel_size=7, stride=2, padding=3, bias=True)
    return model


def get_model(dataset_type, model, pretrain, freeze_pretrain):
    '''
    Return a network model
    :param model:
    :param dataset_type:
    :return:
    '''
    if dataset_type == Mode.GSChannels:
        channels = 8
    elif dataset_type == Mode.RGBChannels:
        channels = 24
    else:
        channels = 3

    return get_resnet_model(channels, model, pretrain, freeze_pretrain)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(LOGGER_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, help="Base path of dataset", required=True)
    parser.add_argument("--dataset", type=Mode, choices=list(Mode), required=True)
    parser.add_argument("--lr", type=float, help="learn rate", default=0.0001)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=10)
    parser.add_argument("--augmentations", type=int, help="Batch size", default=5)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=50)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--create_heatmaps", action="store_true")

    args = parser.parse_args()
    logger.info(f"Dataset path: {args.base_path} , dataset: {args.dataset}")

    width = 256
    height = 256
    pages = 8

    network_type = Network_type.Resnet34

    logger.info(f"Using {network_type}")

    model = get_model(args.dataset, network_type, pretrain=args.pretrain, freeze_pretrain=args.freeze)

    train_dataset = Paper_dataset(args.base_path, args.dataset, width, height, train=True, num_transformations=args.augmentations)
    test_dataset = Paper_dataset(args.base_path, args.dataset, width, height, train=False, num_transformations=0)

    timestamp = time.time()
    trainer = Trainer(train_dataset,
                      test_dataset,
                      logger=logger,
                      pretrained=args.pretrain,
                      freeze=args.freeze,
                      network_type=network_type,
                      dataset_type=args.dataset,
                      log_to_comet=not args.debug,
                      create_heatmaps=args.create_heatmaps)

    trainer.train(model=model,
                  batch_size=args.batch_size,
                  learn_rate=args.lr,
                  epochs=args.epochs,
                  image_type=args.dataset.value)
    logger.info(
        f"Execution time was {time.time() - timestamp} seconds")
