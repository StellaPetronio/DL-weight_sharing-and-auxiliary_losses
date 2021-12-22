"""
Created on Mon Apr 12 17:09:36 2021

@authors: Philippines, Arthur
"""

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

#%% Model 

class ModelNoWeightSharing(nn.Module):

    def __init__(self, n_channels_conv=32, dim_linear1=25, dim_linear2=20, n_layers_linear1=3, n_layers_linear2=3 ,p_dropout=0.1, kernel_size=3):
        """Deep Learning Model to perform two images digit comparison using auxilary loss.
        This network does NOT use weight sharing.

        Parameters
        ---------
        n_channels_conv (int): number of channels in the downsampling convolutional layers
        dim_linear1 (int): dimension (number of neurons) of the FC layers used for the first classifier 
        dim_linear2 (int): dimension (number of neurons) of the FC layers used for the second classifier 
        n_layers_linear1 (int): number of FC layers for classifier 1
        n_layers_linear2 (int): number of FC layers for classifier 2
        p_dropout (float): dropout channel probability
        kernel_size (int): convolutional layer kernel size
        """
        super().__init__()
        self.n_channels_conv = n_channels_conv
        self.dim_linear1 = dim_linear1
        self.dim_linear2 = dim_linear2

        # initial 
        self.initial_conv = nn.Sequential(
                nn.Conv2d(2, self.n_channels_conv, kernel_size=kernel_size),
                nn.BatchNorm2d(n_channels_conv),
                nn.ReLU(),
                nn.Dropout2d(p=p_dropout),
                )


        # each of thise convolution reduces the image size by 2x2.
        self.convnet = nn.Sequential(
                nn.Conv2d(n_channels_conv, n_channels_conv, kernel_size=3),
                nn.BatchNorm2d(n_channels_conv),
                nn.ReLU(),
        )
        
        # Number ordering classifier
        self.classifier1 = nn.Sequential(nn.Linear(4*n_channels_conv, dim_linear1),
                           nn.ReLU(),
                           nn.Sequential( 
                               *(nn.Sequential(nn.Linear(dim_linear1, dim_linear1), nn.ReLU()) for _ in range(n_layers_linear1-2) ) 
                               ),
                           nn.Linear(dim_linear1, 2))

        # Digits class prediction classifier
        self.classifier2 = nn.Sequential(
                nn.Linear(2*n_channels_conv, dim_linear2),
                nn.ReLU(),
                nn.Sequential( 
                    *(nn.Sequential(nn.Linear(dim_linear2, dim_linear2), nn.ReLU()) for _ in range(n_layers_linear2-2) ) 
                    ),
                nn.Linear(dim_linear2, 10)
        )

    
    def forward(self, x):
      # [x]: Cx2x14x14
      x = self.initial_conv(x)
      # [x] Cxn_channelsx12x12
      x = self.convnet(x)
      # [x] 10x10
      x = self.convnet(x)
      # [x] 8x8
      x = self.convnet(x)
      # [x] 6x6
      x = self.convnet(x)
      # [x] 4x4
      x = self.convnet(x)
      # [x] Cx32x2x2

      # now that the convolutions are finished, start the classification 
      # y1: binary classification: [y1] Cx1
      y1 = self.classifier1(x.view(x.size(0), -1))
      # y2: digit of each numbers [y2] Cx2x10
      y2 = self.classifier2(x.view(x.size(0), 2, -1))
      return y1, y2

