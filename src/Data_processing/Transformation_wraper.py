import torchvision

class Transformation_wraper:
    '''Returns a transformation and the normalized original image'''
    def __init__(self, transform, num_transformations):
        mean = [0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        self.num_transformations = num_transformations

        self.transform1 = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(mean=mean, std=std),
                                                          ])
        self.transform2 = torchvision.transforms.Compose([transform,
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(mean=mean, std=std),
                                                          ])

    def __call__(self, item):
        ret = [self.transform1(item)]
        for i in range(self.num_transformations):
            ret.append(self.transform2(item))
        return ret
