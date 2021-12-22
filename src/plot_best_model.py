from dlc_practical_prologue import generate_pair_sets
from matplotlib import pyplot as plt
import numpy as np

import torch 
from get_model import get_model
from load_models import load_models
from framework import *


#%% 

# Device selection
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Load data
train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)
train_input, train_target, train_classes = train_input.to(device), train_target.to(device), train_classes.to(device)
test_input, test_target, test_classes = test_input.to(device), test_target.to(device), test_classes.to(device)

#%% 
#params best model
p_dropout_best, nb_channels_best, nb_lay_1_best, dim_lin1_best, nb_lay_2_best, dim_lin2_best, nb_lay_latent_best, dim_latent_best = 0.5, 32, 3, 25, 3, 25, 3, 8

# Training parameters
batch_sz, lr, nb_epochs, momentum = 100, 1e-1, 100, 0.9
verbose = True

#%% 

#built the best model
best_model =  [["Best model: with weight sharing and with Aux. Loss: ",
                (2, 0.5, p_dropout_best, nb_channels_best, nb_lay_1_best, dim_lin1_best, nb_lay_2_best, dim_lin2_best, nb_lay_latent_best, dim_latent_best), 
               0.5, 
               'Best model', "Best Model"]]

for i, [title, param, w1, label, plot_name] in enumerate(best_model):
    print(title)
    
    model = get_model(param, device)
    
    _, losses, errors, te_losses, te_errors = train_model(model,
                                                            train_input, train_target, train_classes,
                                                            test_input=test_input, test_target=test_target, test_classes=test_classes,
                                                            batch_size=batch_sz,lr=lr, nb_epochs=nb_epochs,
                                                            w1=w1, momentum=momentum,
                                                            plot=True, verbose=verbose)
    
    
    l = losses
    l_test = te_losses
    err = np.array([errors])
    err_te = np.array([te_errors])
    
    # Error rates
    fig, axs = plt.subplots(2, constrained_layout=True)
    plt.rcParams['axes.titley'] = 1.0
    plt.rcParams['axes.titlepad'] = -14
    
    axs[0].plot(range(nb_epochs), err[0,:,0].T, 'c-')
    axs[0].plot(range(nb_epochs), err_te[0,:,0].T, 'c--')
    
    axs[1].plot(range(nb_epochs), err[0,:,1].T, 'c-')
    axs[1].plot(range(nb_epochs), err_te[0,:,1].T, 'c--')

    plt.xlabel('Epochs')
    axs[0].set_ylabel('Error rate in %')
    axs[0].set_title('Digit comparison error rates')
    
    axs[1].set_ylabel('Error rate in %')
    axs[1].set_title('Digit prediction error rates')
    axs[1].legend(['train', 'test'], frameon=False)
    
    plt.suptitle(plot_name)
    plt.show()
    
    fig.savefig("Best_model.png")
