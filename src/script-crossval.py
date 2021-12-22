"""
The goal is to tune the most important parameters of the model using cross validation

How to use the file
-------------------
The first part of the script contains all the hyperparameters to fix and the parameters to use. 
This define the 'experiment' that will be run.
The function `get_model` must be filled accordingly to the definitions of the parameters 
(i.e. using the same convention)

Once the parameters defined and the function completed, the rest of the script should be not changed. 
(in theory)


Parameter definition (that is different for each model)
------------------------------------------------------
- model: 1,2,3 for (No Sharing, Sharing Short, Sharing Long)
- loss weight w: If '1', then no auxilary loss is used. Else, the loss is 
    a linear combination of Loss1 and Loss2
- p dropout
- nb channels convolutions
- n layers linear 1
- dim linear 1
- n layers linear 2
- dim linear 2
(only for sharing models)
- n layers latent
- dim latent


Training parameters to tune
-------------------------
- weights for the loss function

"""
from dlc_practical_prologue import generate_pair_sets
import torch
from framework import * 
from get_model import get_model
import datetime, time
import sys

if __name__ == "__main__":

    # File settings
    dt = datetime.datetime.now()
    file_name = 'results/exp_' + dt.strftime("%d%m%Y_%H%M%S") + ".txt"
    sys.stdout = open(file_name, "w+")

    ##### INPUT OF THE SCRIPT

    # Hyper parameter 
    k_fold = 5
    n_iter_test_val = 10
    nb_epochs = 60
    batch_size = 100

    # # Models to compare
    models = [
        # Model 1: effect of the
        (1, 0.5, 0.5, 32, 3, 25, 3, 25, None , None ), # No Sharing. Auxilary loss
        (1, 0.5, 0.5, 32, 3, 35, 3, 35, None , None ), # No Sharing. Auxilary loss
        (1, 0.5, 0.5, 32, 3, 15, 3, 15, None , None ), # No Sharing. Auxilary loss
        # Model 2: effect of the 'latent' dim
        (2, 0.5, 0.5, 32, 3, 25, 3, 25, 3 , 2 ), # Sharing. Auxilary loss
        (2, 0.5, 0.5, 32, 3, 25, 3, 25, 3 , 8 ), # Sharing. Auxilary loss
        (2, 0.5, 0.5, 32, 3, 25, 3, 25, 5 , 8 ), # Sharing. Auxilary loss
        (2, 0.5, 0.5, 32, 3, 25, 3, 25, 3 , 16 ), # Sharing. Auxilary loss
        # Model 3: same for model 2
        (3, 0.5, 0.5, 32, 3, 25, 3, 25, 3 , 2 ), # Sharing. Auxilary loss
        (3, 0.5, 0.5, 32, 3, 25, 3, 25, 3 , 8 ), # Sharing. Auxilary loss
        (3, 0.5, 0.5, 32, 3, 25, 3, 25, 5 , 8 ), # Sharing. Auxilary loss
        (3, 0.5, 0.5, 32, 3, 25, 3, 25, 3 , 16 ), # Sharing. Auxilary loss
        ]


    ###MODEL FUNCTION  

    ##### SCRIPT
    # (in theory, nothing below this line should be modified)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load data
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)
    train_input, train_target, train_classes = train_input.to(device), train_target.to(device), train_classes.to(device)
    test_input, test_target, test_classes = test_input.to(device), test_target.to(device), test_classes.to(device)


    print("----------------")
    print("CROSS VALIDATION")
    print("----------------")
    print("device = ", device)
    print("model = (nb_channel, n1, dim1, n2, dim2)")
    print("K fold:", k_fold)
    print("N Epochs: ", nb_epochs)

    print("Hyper parameters: \n")
    print("-"*16)
    print("k_fold {}, n_iter {}, nb_epochs {}, batch_size {} \r\n".format(k_fold, n_iter_test_val, nb_epochs, batch_size))


    t1 = time.time()
    model_performances = []

    # iterate for each model 
    for params in models:
        print("")
        print("  New Model ")
        print(params)
        print("-----------------")

        #  Cross validation of this model
        stats = []
        for fold in range(k_fold):
            # a. create the model using the defined parameters
            model = get_model(params = params, device=device)
            w1 = params[1]

            # b. sample training and validation datasets
            N_train = train_input.shape[0]
            shuffling = torch.randperm(N_train)
            training_idcs = shuffling[:int(N_train*0.8)]
            test_idcs = shuffling[int(N_train*0.8):]

            # c. train the model
            train_error = train_model(model, 
                    train_input[training_idcs], 
                    train_target[training_idcs], 
                    train_classes[training_idcs], 
                    batch_size=batch_size, 
                    lr=1e-1, 
                    nb_epochs=nb_epochs, 
                    w1 = w1, 
                    momentum = 0.9,
                    verbose=False)

            # d. assess its performances using boostrapping intervals
            mean_test_error, lower, upper = get_statistics(model, 
                    train_input[test_idcs], 
                    train_target[test_idcs], 
                    train_classes[test_idcs], 
                    n_iter=n_iter_test_val,
                    )
            stats.append(mean_test_error)
            print("Train error: {:.2f} - Test error: {:.2f} [{:.2f} - {:.2f}]".format(train_error, mean_test_error, lower, upper) )

        # Save the performances
        model_performances.append(stats)
    t2 = time.time()

    print("----------")
    print("Elapsed time:", t2-t1)
    print("\nResults")

    #%% Find the best model 
    average_performances = torch.Tensor(model_performances).mean(dim=1)
    best_model_id = average_performances.argmin().item()
    print("    best selected model:", models[best_model_id])
    print("    index: ", best_model_id)
    print("    evaluation on the test dataset")
    model = get_model(models[best_model_id], device)

    # Train it again from scratch with the full dataset
    train_error = train_model(model, 
            train_input, 
            train_target, 
            train_classes, 
            batch_size=batch_size, 
            lr=1e-1, 
            nb_epochs=100, 
            w1 = models[best_model_id][1], 
            momentum = 0.9,
            verbose=True)

    # Assess the results using the full test set 
    mean_test_error, lower, upper = get_statistics(model, 
            test_input, 
            test_target, 
            test_classes, 
            n_iter=100,
            )

    print("Train error: {:.2f} - Test error: {:.2f} [{:.2f} - {:.2f}]".format(train_error, mean_test_error, lower, upper) )

    print("Finished !")

    #%%

    sys.stdout.close()
