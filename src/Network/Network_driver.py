'''
Driver class for the network.
Our "main" method.
'''

import sys
import os
import json

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


def get_resnet_model(number_of_channels, model, pretrain=True):
    # pick resnet 34 and pretrained
    # old resnet 18
    # Copy weights to
    if model != Network_type.Resnet and model != Network_type.Resnet34:
        logger.error(f"Error in get resnet, model is {model}")
        exit()

    if pretrain:
        logger.info(f"Using prertain")
        model = torch.hub.load('pytorch/vision:v0.6.0', model.value, pretrained=True)
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


def get_model(dataset_type, model, pretrain):
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

    return get_resnet_model(channels, model, pretrain)


def coarse_grain_search(args,
                        network_type,
                        width,
                        height,
                        learn_rate_test = True,
                        weight_decay_test = True,
                        batch_size_test = True
                        ):
    
    use_scheduler = False
    debug = True
    pretrain = True
    runs = []
    generic_run = {
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": 1e-6,
        "epochs": args.epochs,
        "use_scheduler": use_scheduler
        }

    if learn_rate_test:
        learn_rates = [10**-i for i in range(4, 9)]
        for lr in learn_rates:
            run = generic_run.copy()
            run['learning_rate'] = lr
            runs.append(run)
    
    if weight_decay_test:
        weight_decays = [10**-i for i in range(6, 9)]
        for weight_decay in weight_decays:
            run = generic_run.copy()
            run['weight_decay'] = weight_decay
            runs.append(run)
            
    if batch_size_test:
        batch_sizes = [10, 25, 50]
        for batch_size in batch_sizes:
            run = generic_run.copy()
            run['batch_size'] = 'batch_size'
            runs.append(run)

    train_dataset = Paper_dataset(args.base_path, args.dataset, width, height, train=True)
    test_dataset = Paper_dataset(args.base_path, args.dataset, width, height, train=False)

    trainer = Trainer(train_dataset,
        test_dataset,
        logger=logger,
        pretrained=args.pretrain,
        network_type=network_type,
        dataset_type=args.dataset,
        log_to_comet=not debug,
        create_heatmaps=args.create_heatmaps)

    for i, run in enumerate(runs):
        print(f"################################ Running run {i}/{len(runs)} ################################")
        print(f"Run {i}: {run}")
        model = get_model(args.dataset, network_type, args.pretrain)
        validation_recall, validation_precision = trainer.train(model=model,
                batch_size=run['batch_size'],
                learn_rate=run['learning_rate'],
                epochs=run['epochs'],
                image_type=args.dataset.value,
                weight_decay=run['weight_decay'],
                use_scheduler=run['use_scheduler'])
        run['run'] = i
        run['validation_recall'] = validation_recall
        run['validation_precision'] = validation_precision
        print("validation_recall: {validation_recall} \n Validation precision: {validation_precision}")
        with open('coarse_grain_results.json', 'w') as json_file:
            json.dump(run, json_file)

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
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--create_heatmaps", action="store_true")
    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--coarse_grain_search", action="store_true")

    args = parser.parse_args()
    logger.info(f"Dataset path: {args.base_path} , dataset: {args.dataset}")

    width = 256
    height = 256
    pages = 8

    network_type = Network_type.Resnet34

    logger.info(f"Using {network_type}")
    
    if args.coarse_grain_search:
        coarse_grain_search(args, network_type, width, height)
    else: 
        model = get_model(args.dataset, network_type, args.pretrain)

        train_dataset = Paper_dataset(args.base_path, args.dataset, width, height, train=True)
        test_dataset = Paper_dataset(args.base_path, args.dataset, width, height, train=False)

        timestamp = time.time()
        trainer = Trainer(train_dataset,
                        test_dataset,
                        logger=logger,
                        pretrained=args.pretrain,
                        network_type=network_type,
                        dataset_type=args.dataset,
                        log_to_comet=not args.debug,
                        create_heatmaps=args.create_heatmaps)

        trainer.train(model=model,
                    batch_size=args.batch_size,
                    learn_rate=args.lr,
                    epochs=args.epochs,
                    image_type=args.dataset.value,
                    use_scheduler=args.use_scheduler)
    logger.info(
        f"Execution time was {time.time() - timestamp} seconds")
