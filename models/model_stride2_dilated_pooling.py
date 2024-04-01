#Total params: 197,152
# train accuracy - 84% (100 epoch) - 79% at 41 epoch
# test accuracy - 88% (100 epoch) - 85% at 41 epoch

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, dropout_value = 0.1):
        super(Net, self).__init__()
        # # Input Block / CONVOLUTION BLOCK 1
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) #input size: 32 x 32 x 3, output size: 32 x 32 x 32, receptive field: 1 + (3-1) * 1 = 3


        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) #input size: 32 x 32 x 32, output size: 32 x 32 x 32, receptive field: 3 + (3-1) * 1 = 5

        ## Strided convolution
        self.SC1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(3, 3), padding=1, stride = 2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) #input size: 32 x 32 x 32, output size: 16 x 16 x 32, receptive field: 5 + (3-1) * 1 = 7
        #Lout =  ((Lin + 2 * padding - dilation * (kernel - 1) - 1)) / stride + 1
        # ((32 + 2 * 1 - 1 * (3 - 1) -1) / 2) + 1 = 16

        # # TRANSITION BLOCK 1 - no transition block

        # CONVOLUTION BLOCK 2
        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) #input size: 16 x 16 x 32, output size: 16 x 16 x 64, receptive field: 7 + (3-1) * 2 = 11

        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) #input size: 16 x 16 x 64, output size: 16 x 16 x 64, receptive field: 11 + (3-1) * 2 = 15

        self.C4_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) #input size: 16 x 16 x 64, output size: 16 x 16 x 64, receptive field: 15 + (3-1) * 2 = 19
      
        ## dilated pooling
        self.SC2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(3, 3), padding=0, dilation = 4, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) #input size: 16 x 16 x 64, output size: 8 x 8 x 64, receptive field: 19 + (9-1) * 2 = 35
        # kernel size for dilated kernel - dilation * (k -1) + 1 = 4 * (3-1) + 1 = 9
        # out size = ((16 + 2 * 0 - 4 * (3 - 1) -1) / 1) + 1 = 8
        
        # TRANSITION BLOCK 2
        self.t2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) #input size: 8 x 8 x 64, output size: 8 x 8 x 32, receptive field: 35 + (1-1)*2 = 35

        # CONVOLUTION BLOCK 3
        
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, groups = 32,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) #input size: 8 x 8 x 32, output size: 8 x 8 x 64, receptive field: 35 + (3-1) * 2 = 39

        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, groups = 64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) #input size: 8 x 8 x 64, output size: 8 x 8 x 64, receptive field: 39 + (3-1) * 2 = 43

        self.C7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, groups = 64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) #input size: 8 x 8 x 64, output size: 8 x 8 x 64, receptive field: 43 + (3-1) * 2 = 47

        ## dilated pooling
        self.SC3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(3, 3), padding=0, dilation = 2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) #input size: 8 x 8 x 64, output size: 4 x 4 x 64, receptive field: 47 + (5-1) * 2 = 55
        # ((8 + 2 * 0 - 2 * (3 - 1) -1) / 1) + 1 = 4
        
        # TRANSITION BLOCK 3
        self.t3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) #input size: 4 x 4 x 64, output size: 4 x 4 x 32, receptive field: 55 + (1-1)*2 = 55

        # CONVOLUTION BLOCK 4

        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, groups = 32,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) #input size: 4 x 4 x 32, output size: 4 x 4 x 64, receptive field: 55 + (3-1) * 2 = 59

        self.C9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, groups = 64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) #input size: 4 x 4 x 64, output size: 4 x 4 x 128, receptive field: 59 + (3-1) * 2 = 63

        self.C10 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, groups = 128,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) #input size: 4 x 4 x 128, output size: 4 x 4 x 128, receptive field: 63 + (3-1) * 2 = 67

        # OUTPUT BLOCK
        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1))
        ) #input size: 4 x 4 x 128, output size: 1 x 1 x 128, receptive field: 67

        self.c11 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10,
                      kernel_size=(1, 1), padding=0, bias=False),
        ) #input size: 1 x 1 x 128, output size: 1 x 1 x 10, receptive field: 67 + (1-1) * 2 =67



    def forward(self, x):
        x = self.C1(x)
        x = x + self.C2(x)

        x = self.SC1(x)
        # x = self.t1(x)

        x = self.C3(x)
        x = self.C4(x)
        x = x + self.C4_1(x)

        x = self.SC2(x)
        x = self.t2(x)

        x = self.C5(x)
        x = self.C6(x)
        x = x + self.C7(x)

        x = self.SC3(x)
        x = self.t3(x)

        x = self.C8(x)
        x = self.C9(x)
        x = self.C10(x)

        x = self.GAP(x)
        x = self.c11(x)

        x = x.squeeze()

        return F.log_softmax(x, dim=-1)
