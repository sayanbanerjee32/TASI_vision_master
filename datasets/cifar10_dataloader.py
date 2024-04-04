import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .cifar10_dataset import Cifar10Dataset

def get_train_test_transforms(is_train = True, mean = (0.4914,0.4822,0.4465), std = (0.247,0.243,0.262)):
    # Train Phase transformations
    if is_train:
        transforms = A.Compose([
                                    A.Compose([
                                        A.PadIfNeeded (min_height=40, min_width=40, p = 1.0),
                                        A.RandomCrop(p=1, height=32, width=32),
                                    ], p = 0.5),
                                    A.HorizontalFlip(p=0.5),
                                    # A.ShiftScaleRotate(p=0.5),
                                    A.Normalize(mean, std),
                                    A.Compose([A.PadIfNeeded (min_height=48, min_width=48, p = 1.0),
                                                A.CoarseDropout(max_holes = 1, max_height=8, max_width=8, 
                                                                min_holes = 1, min_height=8, min_width=8,
                                                    fill_value=[0.4914,0.4822,0.4465], mask_fill_value = None, p=1),
                                            A.CenterCrop(height=32, width=32, p=1),
                                    ], p = 0.5),
                                    ToTensorV2(),
                                    ])

    else:
        transforms = A.Compose([
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])
    
    return transforms

def get_cifar10_data_loader(dataloader_args,is_train = True):
    transforms = get_train_test_transforms(is_train = is_train)
    data = Cifar10Dataset(train=is_train, transform= transforms)
    
    # dataloader
    return torch.utils.data.DataLoader(data, **dataloader_args)

    
