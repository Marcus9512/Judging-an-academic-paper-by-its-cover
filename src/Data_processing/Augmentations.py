import torch
import torchvision
import numpy as np
import cv2
from PIL import Image

mean = [0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

class Augmentations():
    def __init__(self, probs,height, width, num_trans=10):
        self.probs = probs
        self.height = height
        self.width = width
        self.num_trans = num_trans

    #https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/2
    class add_gaussian_noise(object):
        def __init__(self, mean=0., std=1.):
            self.std = std
            self.mean = mean
            
        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        
        def __repr__(self):
            return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



    class GaussianBlur():
        def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
            self.sigma_min = sigma_min
            self.sigma_max = sigma_max
            self.kernel_size = kernel_size

        def __call__(self, img):
            sigma = np.random.uniform(self.sigma_min, self.sigma_max)
            img = cv2.GaussianBlur(np.array(img), (self.kernel_size, self.kernel_size), sigma)
            return Image.fromarray(img.astype(np.uint8))


    def get_transform(self):
        np.random.seed(34)
        random_transform = np.random.choice(self.num_trans, p=self.probs)
        if(random_transform == 0):
            return self.get_normalisation()

        if(random_transform == 1):
            return self.get_resize_crop()
            
        if(random_transform == 2):
            return self.get_vertical_flip

        if(random_transform == 3):
            return self.get_gaussian_noise()

        if(random_transform == 4):
            ##Wtf is kernel size
            return self.get_gaussian_blur()
                 
        if(random_transform == 5):
            return self.get_colorjitter()
        
        



    def get_normalisation(self):
        return torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=mean, std=std),
                                                        ])

    def get_resize_crop(self):
        return torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(size=(int(self.height*2), int(self.width*4)),
                                                    scale=(0.9, 1.0)),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=mean, std=std),
                                            ])
    def get_vertical_flip(self):
        return torchvision.transforms.Compose([torchvision.transforms.RandomVerticalFlip(p=1),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=mean, std=std),
                                            ])
    def get_gaussian_noise(self):
        return torchvision.transforms.Compose([torchvision.transforms.RandomVerticalFlip(p=1),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=mean, std=std),
                                            ])
    def get_gaussian_blur(self):
        return torchvision.transforms.Compose([self.GaussianBlur(kernel_size=3),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=mean, std=std),
                                                        ])
    
    def get_colorjitter(self):
        return torchvision.transforms.Compose([torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=1, hue=[-0.5,0.5]),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=mean, std=std),
                                    ])


    
    


    