import torchvision
import numpy as np
import albumentations as A
import cv2

def get_transform(probs,height, width, num_trans=10):
    np.random.seed(34)
    random_transform = np.random.choice(num_trans, p=probs)
    mean = [0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    if(random_transform == 0):
        return torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(mean=mean, std=std),
                                                    ])
    if(random_transform == 1):
        return torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(size=(int(height*2), int(width*4)),
                                                    scale=(0.9, 1.0)),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(mean=mean, std=std),
                                               ])
        
        
        
    if(random_transform == 2):
        return torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(p=1),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(mean=mean, std=std),
                                               ])
     
    if(random_transform == 3):
        return torchvision.transforms.Compose([torchvision.transforms.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=1),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(mean=mean, std=std),
                                               ])
    if(random_number == 4):
        ##No idea if this works
        return torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda x : x + torch.randn_like(x)),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(mean=mean, std=std),
                                               ])
        
    if(random_number == 5):
    if(random_number == 6):

    if(random_number == 7):

    if(random_number == 8):

    if(random_number == 9):

   


    