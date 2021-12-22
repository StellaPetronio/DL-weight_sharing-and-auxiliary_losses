from dlc_practical_prologue import generate_pair_sets
import time

import torch 
from get_model import get_model
from load_models import load_models
from framework import *

# Device selection
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Load data
train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)
train_input, train_target, train_classes = train_input.to(device), train_target.to(device), train_classes.to(device)
test_input, test_target, test_classes = test_input.to(device), test_target.to(device), test_classes.to(device)

# get the best model
params = (2, 0.5, 0.1, 32, 3, 25, 3, 25, 3 , 8)
model = get_model(params, device)

# Train the models
t1 = time.time()
train_model(model, train_input, train_target, train_classes, batch_size=100, lr=1e-1, nb_epochs=100, w1 = 0.5, momentum=0.9)
t2 = time.time()
print("Elapsed time:", t2-t1)

#%% evaluate the model
print("Evaluate model ")
evaluate_model(model, test_input, test_target, test_classes, w1=1)

# Confidence interval of model's performance
mean, lower, upper = get_statistics(model, test_input, test_target, test_classes)
print("Statistics: mean test error :{}%, [{}%, {}%]".format(mean, lower, upper))




