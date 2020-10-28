'''
Driver class for the network.
Our "main" method.
'''

from src.Tools.Tools import *
from src.Network.Trainer import *
from src.Tools.open_review_dataset import *

LOGGER_NAME = "Network_Driver"

class Network_type(Enum):
    Resnet = "resnet"

    def __str__(self):
        return self.value

def sanity_check_paper_dataset(dataset_path):
    '''
    Sanity check for the Paper_dataset class
    :param dataset_path:
    :return:
    '''
    dataset = Paper_dataset(dataset_path, print_csv=True)

    for i in range(0, dataset.len):
        info = dataset[i]
        print(info["title"])

    info = dataset[2]
    print_image_from_array(info["image"] * 255)


def get_resnet_model(number_of_channels):
    #pick resnet 34 and pretrained
    #Copy weights to
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True, num_classes=1)
    # modifying the input layer to accept 8 channels input:
    model.conv1 = nn.Conv2d(number_of_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
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

def convert_mode_to_str(mode):
    if mode == Mode.RGBFrontPage:
        return "rgbfrontpage"
    elif mode == Mode.GSFrontPage:
        return "gsfrontpage"
    elif mode == Mode.GSChannels:
        return "gschannels"
    elif mode == Mode.RGBChannels:
        return "rgbchannels"
    elif mode == Mode.RGBBigImage:
        return "rgbbigimage"
    elif mode == Mode.GSBigImage:
        return "gsbigimage"
    return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(LOGGER_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, help="Base path of dataset", required=True)
    parser.add_argument("--dataset", type=Mode, help="bigimage, rgbchannels or gschannels", required=True)

    parser.add_argument("--lr", type=float, help="learn rate", default=0.0001)
    parser.add_argument("--lr_decay", type=float, help="learn decay", default=1e-9)
    parser.add_argument("--lr_momentum", type=float, help="learn momentum", default=1e-9)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=10)

    parser.add_argument("--epochs", type=int, help="Number of epochs", default=50)

    args = parser.parse_args()
    logger.info(f"Dataset path: {args.base_path} , dataset: {args.dataset}")

    width = 256
    height = 256
    pages = 8

    network_type = Network_type.Resnet
    model = get_model(args.dataset, Network_type.Resnet)

    timestamp = time.time()
    trainer = Trainer(Paper_dataset(args.base_path, args.dataset, width, height), logger=logger,
                      network_type=network_type, dataset_type=args.dataset)

    trainer.train(model=model, batch_size=args.batch_size, learn_rate=args.lr, learn_decay=args.lr_decay,
                  learn_momentum=args.lr_momentum, epochs=args.epochs, image_type=convert_mode_to_str(args.dataset))
    logger.info(
        f"Execution time was {time.time() - timestamp} seconds")

    # SAVED INFORMATION

    # Having modified the last layer, see:
    # https://github.com/pytorch/vision/blob/21153802a3086558e9385788956b0f2808b50e51/torchvision/models/resnet.py#L99
    # &&
    # https://github.com/pytorch/vision/blob/21153802a3086558e9385788956b0f2808b50e51/torchvision/models/resnet.py#L167

    # potential fix if above does not work:

    """
    model.fc = nn.Linear(512 * 1000, 1)             # where 512 * 1000 = input nodes, 1 = num_classes
    """
    # other resnet implementations:
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=False)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=False)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=False)

'''
Code graveyard

def generate_dataset(args, width, height, num_pages):

    generated = False
    #Generate dataset
    if args.dataset == Mode.GSChannels:
        convert_pdf_dataset(dataset_base_path=ensure_non_null("base_path", args.base_path),
                            num_pages=ensure_non_null("num_pages", num_pages),
                            width=ensure_non_null("image_width", width),
                            height=ensure_non_null("image_height", height),
                            mode=Mode.GSChannels)
        generated = True
    elif args.dataset == Mode.RGBChannels:
        convert_pdf_dataset(dataset_base_path=ensure_non_null("base_path", args.base_path),
                            num_pages=ensure_non_null("num_pages", num_pages),
                            width=ensure_non_null("image_width", width),
                            height=ensure_non_null("image_height", height),
                            mode=Mode.RGBChannels)
        generated = True
    elif args.dataset == Mode.BigImage:
        convert_pdf_dataset(dataset_base_path=ensure_non_null("base_path", args.base_path),
                            num_pages=ensure_non_null("num_pages", num_pages),
                            width=ensure_non_null("image_width", width),
                            height=ensure_non_null("image_height", height),
                            mode=Mode.BigImage)
        generated = True

    return generated
    
///// main
    parser.add_argument("--generate", type=str, help="Set to yes if data should be generated otherwise no", required=True)
    if args.generate == "yes":
        status = generate_dataset(args, width, height, pages)
        if not status:
            logger.warn(f"No data generated")
            exit(-1)

'''