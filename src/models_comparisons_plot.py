"""
The purpose of this script is to compare model performances with plots

5 models are compared:
(1) Baseline model without weight sharing and auxiliary loss (model_no_sharing)
(2) Model without weight sharing but trained with an auxiliary loss (model_no_sharing)
(4) Model with weight sharing and no auxiliary loss, and 3 convolutional layers (model_weight_sharing)
(5) Model with weight sharing and auxiliary loss, 3 convolutional layers (best architecture derived from model_weight_sharing)
(6) Model 5 with cross validated hyper parameters, and our best model so far

Each of these models are trained over 40 epochs, and with a batch size of 100.
Test and train error rates for digit comparison and digit class prediction are reported.

"""

from dlc_practical_prologue import generate_pair_sets
from matplotlib import pyplot as plt
import time
import torch 
from get_model import get_model
from load_models import load_models
from framework import *
import numpy as np

########################################################################################################################
# Device selection
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Load data
train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)
train_input, train_target, train_classes = train_input.to(device), train_target.to(device), train_classes.to(device)
test_input, test_target, test_classes = test_input.to(device), test_target.to(device), test_classes.to(device)

########################################################################################################################

print("-"*40)
print("MODELS ARCHITECTURES")
print("-"*40)

# Training parameters
batch_sz, lr, nb_epochs, momentum = 100, 1e-1, 70, 0.9
verbose = True

########################################################################################################################

print("Impact of auxiliary loss and weight sharing on model prediction of digits comparison and class prediction: ")

# Load the models to compare
model_comparisons = load_models()

# Error rates
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

print(model_comparisons)
for i, [label, c, param, w1] in enumerate(model_comparisons): 

    mdl = get_model(param, device)

    _, _, errors, _, te_errors = train_model(mdl,
                                            train_input, train_target, train_classes,
                                            test_input=test_input, test_target=test_target, test_classes=test_classes,
                                            batch_size=batch_sz,lr=lr, nb_epochs=nb_epochs,
                                            w1=w1, momentum=momentum,
                                            plot=True, verbose=verbose)

    errors, te_errors = np.array(errors), np.array(te_errors)
    c_test = c+'-'

    #ax1.plot(range(nb_epochs), errors[:,0], c, label=label)
    ax1.plot(range(nb_epochs), te_errors[:, 0], c_test, label=label + ' test')

    #ax2.plot(range(nb_epochs), errors[:,1], c, label=label)
    ax2.plot(range(nb_epochs), te_errors[:,1], c_test, label=label + ' test')


ax1.set_xlabel('Epochs')
ax2.set_xlabel('Epochs')
ax1.set_ylabel('Error rate in %')
ax1.set_title('Digit comparison error rates')
ax2.set_ylabel('Error rate in %')
ax2.set_title('Digit prediction error rates')
#ax1.legend()
ax2.legend()
plt.show()


########################################################################################################################













