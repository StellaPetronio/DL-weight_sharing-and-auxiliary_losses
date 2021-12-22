import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from .evaluation import evaluate_model

def shuffled_data(x, y_target, y_classes, shuffle = True):
    """
    Shuffles dataset for stochastic gradient descent
    Parameters:
        - x (torch.Tensor): of dim (Nx2x14x14) input
        - y (torch.Tensor): of dim(Nx2) target
        - shuffle (bool): if set to true, shuffle data
    Returns:
        - shuffled_x (torch.Tensor): shuffled input
        - shuffled_y (torch.Tensor): shuffled target
    """
    if shuffle:
        n = x.shape[0]
        shuffled_ind = torch.randperm(n)
        shuffled_x = x[shuffled_ind, :,:,:]
        shuffled_y_target = y_target[shuffled_ind]
        shuffled_y_classes = y_classes[shuffled_ind]
        return shuffled_x, shuffled_y_target, shuffled_y_classes
    else:
        return x, y_target, y_classes


def train_model(model, train_input, train_target, train_classes, test_input=None, test_target=None, test_classes=None, batch_size=10, lr=1e-1, nb_epochs=50, w1 = 0.5, momentum=0.9, shuffle = True, verbose=True, plot=False):
    """
    Function to train a Model. 
    
    Parameters (to be hand tuned)
    ----------
    - batch_size
    - lr: learning rate
    - nb_epochs
    - w1: weights attributed to Loss1 (if 1, no auxilary loss is used)
    - momentum: optimizer momentum
    - shuffle: True if train data has to be shuffled
    """
    w2 = 1 - w1 
    N_train = train_input.size(0)

    # reset the parameters of the model (just to make sure !)
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    # Loss object
    criterion1 = nn.CrossEntropyLoss() # auxiliary loss
    criterion2a = nn.CrossEntropyLoss() # main classification loss a
    criterion2b = nn.CrossEntropyLoss() # main classification loss b

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum = momentum)

    # Lists for convenient plotting
    errors, losses = [], []
    te_errors, te_losses = [], []

    # Train the model
    model.train(True)
    for e in range(nb_epochs):
        acc_loss, nb_errors1, nb_errors2 = 0, 0, 0

        # Shuffle data
        train_input, train_target, train_classes = shuffled_data(train_input, train_target, train_classes, shuffle = shuffle)

        # Looping over the mini-batches (training set, training classes, training label of number ordering)
        for b in range(0, N_train, batch_size):
            x_train, y_train1, y_train2 = train_input[b:b+batch_size], train_target[b:b+batch_size], train_classes[b:b+batch_size]
            # compute the predictions
            y1, y2 = model(x_train) # y1 : digits comparison, y2 : digits class prediction
            # losses computation (3 terms to compute)(3 terms to compute)
            loss1 = criterion1(y1, y_train1)
            loss2a = criterion2a(y2[:,0], y_train2[:,0])
            loss2b = criterion2b(y2[:,1], y_train2[:,1])
            loss = w1 * loss1 + w2 * (loss2a + loss2b)
            acc_loss = acc_loss + loss.item()
            # backward pass
            model.zero_grad()
            loss.backward()
            optimizer.step()
            # update training error
            nb_errors1 += (y1.argmax(dim=1) != y_train1).sum()
            nb_errors2 += (y2.argmax(dim=2) != y_train2).sum()

        errors.append([nb_errors1 / N_train * 100, nb_errors2 / 2 / N_train * 100])
        losses.append(acc_loss)

        # Evaluate model on test set
        if plot and test_target is not None and test_input is not None and test_classes is not None :
            te_loss, te_error1, te_error2 = evaluate_model(model, test_input, test_target, test_classes,
                                                           batch_size=batch_size, w1=w1, verbose=False)

            te_losses.append(te_loss)
            te_errors.append([te_error1, te_error2])

        if verbose and e % 5 == 0:
            print("Epoch n° {}, Loss = {:.2f}, Training errors = [{:.2f} - {:.2f}] %".format(e, acc_loss, 100*nb_errors1 / N_train, 100*nb_errors2 / 2 / N_train))

    model.train(False)

    if plot:
        return 100 * nb_errors1 / N_train, losses, errors, te_losses, te_errors

    return 100 * nb_errors1 / N_train


