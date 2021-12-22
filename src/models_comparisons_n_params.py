import torch
from torchsummary import summary
from get_model import get_model
import time

# Device selection
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
#parameters for the models
p_dropout, nb_channels, nb_lay_1, dim_lin1, nb_lay_2, dim_lin2, nb_lay_latent, dim_latent, kernel_sz = 0.1, 32, 4, 25, 4, 25, 4, 2, 3

#parameters best model
p_dropout_best, nb_channels_best, nb_lay_1_best, dim_lin1_best, nb_lay_2_best, dim_lin2_best, nb_lay_latent_best, dim_latent_best = 0.5, 32, 3, 25, 3, 25, 3, 8

###################################################ù#####################################################################

model_comparisons = [ 
    ["1. Model without Aux. Loss: ",
    (1, 1, p_dropout, nb_channels, nb_lay_1, dim_lin1, nb_lay_2, dim_lin2, None, None), 1],
                     
    ["2. Model with Aux. Loss: ",
    (1, 0.5, p_dropout, nb_channels, nb_lay_1, dim_lin1, nb_lay_2, dim_lin2, None, None), 0.5],
                     
    ["3. Model 3 convolutions layers with weight sharing but without Aux. Loss: ",
    (3, 1, p_dropout, nb_channels, nb_lay_1, dim_lin1, nb_lay_2, dim_lin2, nb_lay_latent, dim_latent), 1],
                      
    ["4. Model 2 convolutions layers with weight sharing but without Aux. Loss: ",
    (2, 1, p_dropout, nb_channels, nb_lay_1, dim_lin1, nb_lay_2, dim_lin2, nb_lay_latent, dim_latent), 1],
                      
    ["5. Model with weight sharing and with Aux. Loss: ",
    (2, 0.5, p_dropout, nb_channels, nb_lay_1, dim_lin1, nb_lay_2, dim_lin2, nb_lay_latent, 2), 1],
                     
    ["6. (Best Model:) Model with weight sharing and with Aux. Loss: ",
    (2, 0.5, p_dropout_best, nb_channels_best, nb_lay_1_best, dim_lin1_best, nb_lay_2_best, dim_lin2_best, nb_lay_latent_best, dim_latent_best), 0.5]
                       
    ]

for i, [title, param, w1] in enumerate(model_comparisons): 
    print(title)
    model = get_model(param,device)
    summary(model, (2,14,14))
