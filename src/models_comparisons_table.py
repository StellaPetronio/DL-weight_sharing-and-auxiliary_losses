"""
The purpose of this script is to compare model performances to create the table of the report
6 models are compared:
(1) Baseline model without weight sharing and auxiliary loss (model_no_sharing)
(2) Model without weight sharing but trained with an auxiliary loss (model_no_sharing)
(3) Model with weight sharing and no auxiliary loss, and 3 convolutional layers (model_weight_sharing)
(4) Model with weight sharing and no auxiliary loss, and 2 convolutional layers (model_weight_sharing)
(5) Model with weight sharing and auxiliary loss, 2 convolutional layers (best architecture derived from model_weight_sharing)
(6) Model with cross validated hyper parameters, our best model

Each of these models are trained over 25 epochs, and with a batch size of 100.
The mean and the standard deviation of the test error rates for digit comparison and digit class prediction and the mean time needed for triainig a model over a 10 times are reported in the table of the report.
"""


from dlc_practical_prologue import generate_pair_sets
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

# Load the models to compare
model_comparisons = load_models()

# Training parameters
batch_sz, lr, nb_epochs, momentum = 100, 1e-1, 100, 0.9
verbose = False
    
for i, [title, color, param, w1] in enumerate(model_comparisons):
    test_error_1 = [] #digits comparison
    test_error_2 = [] #digits class prediction
    time_ = [] #time needed to train
    print("--------------------------------")
    print("Start evaluation for model: ", i)
    print("--------------------------------")
    for j in range(10):
        print("Iteration number:", j)
        model = get_model(param,device)
        
        # Train the models
        t1 = time.time()
        train_errors = train_model(model,
                train_input, 
                train_target, 
                train_classes,
                test_input=test_input, 
                test_target=test_target, 
                test_classes=test_classes, 
                batch_size=batch_sz,
                lr=lr, 
                nb_epochs=nb_epochs,
                w1=w1,
                momentum=momentum,
                plot=False,
                verbose=verbose)
        t2 = time.time()
       
        # Evaluate the models
        loss, te_error1, te_error2 = evaluate_model(model, 
                test_input, 
                test_target, 
                test_classes, 
                batch_size=batch_sz, 
                w1=w1, 
                verbose=True)
    
        time_.append(t2-t1)
        test_error_1.append(te_error1.item())
        test_error_2.append(te_error2.item())
        
    print("------------------------")
    print("Result for model: ", i)
    print("------------------------")
    print(title)
    mean_test_error_1 = np.mean(test_error_1, dtype = np.float32)
    std_test_error_1 = np.std(test_error_1, dtype = np.float32)
    mean_test_error_2 = np.mean(test_error_2, dtype = np.float32)
    std_test_error_2 = np.std(test_error_2, dtype = np.float32)
    mean_time_ = np.mean(time_, dtype = np.float32)

    print("Test Error digit comparison: [" , mean_test_error_1, "  ±  ", std_test_error_1, " ]")
    print("Test Error class prediction: [" , mean_test_error_2, "  ±  ", std_test_error_2, " ]")
    print("Time of training: ", mean_time_)






