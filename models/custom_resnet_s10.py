import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, dropout_value = 0.1):
        super(Net, self).__init__()
        # # Input Block / CONVOLUTION BLOCK 1
        # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        self.PrepLayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 


        # Layer 1
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
        # Add(X, R1)
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 

        self.R1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout(dropout_value)
        )

        # Layer 2 -
        # Conv 3x3 [256k]
        # MaxPooling2D
        # BN
        # ReLU

        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Dropout(dropout_value)
        )

        # Layer 3 -
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        # R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        # Add(X, R2)
        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 

        self.R2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.Dropout(dropout_value)
        )

        # MaxPooling with Kernel Size 4
        self.p = nn.MaxPool2d(kernel_size=4, stride=4)

        #FC Layer 
        self.c4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10,
                      kernel_size=(1, 1), padding=0, bias=False),
        ) 



    def forward(self, x):
        x = self.PrepLayer(x)

        x = self.C1(x)
        x = x + self.R1(x)

        x = self.C2(x)

        x = self.C3(x)
        x = x + self.R2(x)

        x = self.p(x)
 
        x = self.c4(x)

        x = x.squeeze()

        return x
