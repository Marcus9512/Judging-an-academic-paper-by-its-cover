from PIL import Image
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform



class saveFeatures():
    features = None
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()

    def remove(self):
        self.hook.remove()


def get_CAM(feature_convolution, weights, class_index):
    _, nc, h, w = feature_convolution.shape
    cam = weights[class_index].dot(feature_convolution.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]


if __name__ == "__main__":
    # load model:
    model.load_state_dict(torch.load(model_path))
    
    # image = load "random" input data
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, help="Base path of dataset", required=True)

    args = parser.parse_args()

    image = np.load(self.data_path + "/" + data["paper_path"] + self.dataset_type + self.res+".npy")

    image = image / 255
    
    # This line might be needed by pytorch to switch place for the channel data
    image = image.transpose((2, 0, 1))



    model.cuda()
    model.eval()

    # setup hook to get last convolutional layer
    final_layer = model._modules.get('layer4')
    activated_features = saveFeatures(final_layer)

    # make prediction
    prediction = model(image)
    pred_probabilities = F.softmax(prediction).data.squeeze()
    activated_features.remove()

    # get parameters to create CAM
    weight_softmax_params = list(model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

    # get prediction from network (probably not necessary with our implementation - should work without this)
    class_idx = topk(pred_probabilities,1)[1].int()

    # create heatmap overlay
    heatmap = getCAM(activated_features.features, weight_softmax, class_idx)

    # show image + overlay
    imshow(display_transform(image))
    imshow(skimage.transform.resize(heatmap[0], tensor.shape[1:3]), alpha=0.5, cmap='jet');
