import numpy as np
import torch
import torch.nn as nn

#%% Testing the network

def evaluate_model(model, test_input, test_target, test_classes, batch_size=100, w1=0.5, verbose=True):
    """Given a model and all the required inputs, it will return the performances of this model 
    on the provided inputs. 
    """
    N_test = test_input.size(0)
    model.eval()

    nb_errors1, nb_errors2 = 0, 0
    # Loss object
    criterion1 = nn.CrossEntropyLoss() # auxiliary loss
    criterion2a = nn.CrossEntropyLoss() # main classification loss a
    criterion2b = nn.CrossEntropyLoss() # main classification loss b

    # Batches
    with torch.no_grad():
        for b in range(0, N_test, batch_size):
            x_test, y_test1, y_test2 = test_input[b:b+batch_size], test_target[b:b+batch_size], test_classes[b:b+batch_size]
            # compute the predictions
            y1, y2 = model(x_test) # y1 : digits comparison, y2 : digits class prediction
            # losses computation (3 terms to compute)(3 terms to compute)
            loss1 = criterion1(y1, y_test1)
            loss2a = criterion2a(y2[:,0], y_test2[:,0])
            loss2b = criterion2b(y2[:,1], y_test2[:,1])
            # update test error
            nb_errors1 += (y1.argmax(dim=1) != y_test1).sum()
            nb_errors2 += (y2.argmax(dim=2) != y_test2).sum()

    if verbose:
        print("Test errors = [{:.2f} - {}] %".format(100 * nb_errors1 / N_test, 100 * nb_errors2 / (2* N_test)))

    model.train()
    return (w1*loss1 + (1-w1)*(loss2a + loss2b), 100 * nb_errors1 / N_test, 100 * nb_errors2 / (2* N_test))

#%% Function to assess bootstrapping confidence interval

def get_statistics(model, test_input, test_target, test_classes, n_iter=100, sampling_size=1000):
    """Get bootstrappping confidence interval on the evaluation of a DL. 
    This function is used for cross-validation, but it is not used for model evaluation. 
    """
    # start the boostrapring
    errs = []
    for iter in range(n_iter):
        # bootstrap a random sample of the dataset
        idcs = np.random.choice(np.arange(test_input.size(0)), size=sampling_size)
        X_test = test_input[idcs]
        Y_test_target = test_target[idcs]
        Y_test_classes = test_classes[idcs]

        # compute performances
        loss, acc1, acc2 = evaluate_model(model, X_test, Y_test_target, Y_test_classes, verbose=False)
        errs.append(acc1.item())

    # average them and get the bootstrap confidence interval
    mean = np.mean(errs)
    lower = np.percentile(errs, 5)
    upper = np.percentile(errs, 95)
    return mean, lower, upper

