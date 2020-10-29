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
    Resnet = "resnet"

    def __str__(self):
        return self.value


def get_resnet_model(number_of_channels):
    # pick resnet 34 and pretrained
    # Copy weights to
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False, num_classes=1)
    # modifying the input layer to accept 8 channels input:
    model.conv1 = nn.Conv2d(number_of_channels, 64, kernel_size=7, stride=2, padding=3, bias=True)
    return model


def get_model(dataset_type, model):
    '''
    Return a network model
    :param model:
    :param dataset_type:
    :return:
    '''
    if dataset_type == Mode.GSChannels:
        channels = 8
    else:
        channels = 3

    if model == Network_type.Resnet:
        return get_resnet_model(channels)
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(LOGGER_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, help="Base path of dataset", required=True)
    parser.add_argument("--dataset", type=Mode, choices=list(Mode), required=True)
    parser.add_argument("--lr", type=float, help="learn rate", default=0.0001)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=10)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=50)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--create_heatmaps", action="store_true")

    args = parser.parse_args()
    logger.info(f"Dataset path: {args.base_path} , dataset: {args.dataset}")

    width = 256
    height = 256
    pages = 8

    network_type = Network_type.Resnet
    
    model = get_model(args.dataset, Network_type.Resnet)

    timestamp = time.time()
    trainer = Trainer(Paper_dataset(args.base_path, args.dataset, width, height), logger=logger,
                      network_type=network_type, dataset_type=args.dataset, log_to_comet=not args.debug, create_heatmaps=args.create_heatmaps)

    trainer.train(model=model,
                  batch_size=args.batch_size,
                  learn_rate=args.lr,
                  epochs=args.epochs,
                  image_type=args.dataset.value)
    logger.info(
        f"Execution time was {time.time() - timestamp} seconds")
