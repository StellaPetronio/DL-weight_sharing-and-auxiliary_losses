import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

#%% Model Weight Sharing

class ModelWeightSharing(nn.Module):
    
    def __init__(self, use_long = False, nb_channels = 32, dim_linear1 = 25, dim_linear2 = 25, n_layers_latent = 4, n_layers_linear1 = 4, n_layers_linear2 = 4, dim_latent=2, p = 0.1):
        super().__init__()
        
        # compute the latent dimension
        self.latent_input_dim = (-1,nb_channels * 8) if not use_long else (-1, nb_channels*2*3*3)

        if use_long:
            self.convs = nn.Sequential(nn.Conv2d(1,nb_channels, kernel_size = 3, padding = 1), # N x 32 x 12 x 12
                                         nn.MaxPool2d(kernel_size = 2), # N x 32 x 6 x 6 
                                         nn.BatchNorm2d(nb_channels),
                                         nn.ReLU(),
                                         nn.Dropout2d(p=p),
                                         nn.Conv2d(nb_channels, nb_channels * 2, kernel_size = 3, padding = 1), # N x 64 x 4 x 4
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         nn.Dropout2d(p=p),
                                         nn.Conv2d(nb_channels * 2, nb_channels*2, kernel_size = 3, padding = 1), # N x 64 x 2 x 2
                                         nn.MaxPool2d(kernel_size = 2), # N x 64 x 1 x 1
                                         nn.BatchNorm2d(nb_channels * 2),
                                         nn.ReLU())

            self.classifier_latent = nn.Sequential(nn.Linear(nb_channels*2*3*3,dim_linear1),
                                                   nn.ReLU(),
                                                   nn.Sequential(*(nn.Sequential(nn.Linear(dim_linear1, dim_linear1), nn.ReLU()) for _ in
                                                 range(n_layers_latent-2))),
                                                   nn.Linear(dim_linear1, dim_latent))
        else: 
             self.convs = nn.Sequential(nn.Conv2d(1,nb_channels,kernel_size=5), # N x 32 x 10 x 10
                                       nn.MaxPool2d(kernel_size=2), # N x 32 x 5 x 5
                                       nn.BatchNorm2d(nb_channels),
                                       nn.ReLU(),
                                       nn.Dropout2d(p=p),
                                       nn.Conv2d(nb_channels,nb_channels*2,kernel_size=2), # N x 64 x 4 x 4
                                       nn.MaxPool2d(kernel_size=2), # N x 64 x 2 x 2
                                       nn.BatchNorm2d(nb_channels*2))

             self.classifier_latent = nn.Sequential(nn.Linear(8*nb_channels, dim_linear1),
                                                    nn.ReLU(),
                                                    *(nn.Sequential(nn.Linear(dim_linear1, dim_linear1), nn.ReLU()) for _ in
                                                 range(n_layers_latent-2)),
                                                    nn.Linear(dim_linear1, dim_latent))

        # Digits comparison classifier
        self.classifier1 = nn.Sequential(nn.Linear(2 * dim_latent, dim_linear2), nn.ReLU(),
                                         nn.Sequential(*(nn.Sequential(nn.Linear(dim_linear2, dim_linear2), nn.ReLU()) for _ in range(n_layers_linear2-2))),
                                         nn.Linear(dim_linear2, 2))

        # Digits class prediction classifier
        self.classifier2 = nn.Sequential(
                 nn.Linear(dim_latent, dim_linear2),
                 nn.ReLU(),
                 nn.Sequential( 
                     *(nn.Sequential(nn.Linear(dim_linear2, dim_linear2), nn.ReLU()) for _ in range(n_layers_linear2-2) ) 
                     ),
                 nn.Linear(dim_linear2, 10)
         )

    def forward_(self, x):
        y = self.convs(x)
        y = self.classifier_latent(y.view(self.latent_input_dim))
        return y
      
    def forward(self,x):
        x1, x2 = x.split(split_size=1, dim=1)
        #forward pass for digit x1
        x1 = self.forward_(x1)
        #forward pass for digit x2
        x2 = self.forward_(x2)
        
        # concatenate the two digit after forward pass
        x = torch.cat((x1,x2), dim = 1)
        y1 = self.classifier1(x) # comparison classifier
        y2 = self.classifier2(x.view(x.size(0), 2, -1)) # class prediction classifier
    
        return y1, y2

