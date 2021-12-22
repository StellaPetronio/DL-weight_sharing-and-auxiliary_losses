# Project 1

_Arthur Bricq, Stella Petronio and Philippine des Courtils_

This project aims at exploring several Deep Network architectures, especially ones including weight sharing and auxiliary losses, on a specific classification task. This task consists in predicting the greatest number out of two MNIST digits.

## Table of contents
* [Folder architecture](#folder-architecture)
* [Usage](#usage)
    - [Cross Validation](#cross-validation)
    - [Training](#usage)
    - [Evaluation](#evaluation)
    - [Plots](#plots)

## Folder architecture

The working code is on the **src/ directory** and contains the following files:
- framework/ 
    - evaluation.py: functions to evaluate a model 
    - train.py: functions to train a model 
    - model_no_sharing.py: Class of model without weight sharing
    - model_weight_sharing: Class of models with weight sharing
- models_comparisons_n_params: compare the models in term of number of parameters
- models_comparisons_plot: make plots for the report
- models_cmparisons_table: give reslts for the report's table
- plot_best_model: train curve for the best model 
- script-crossval.py: run a cross validation of several specified models 
- script-train.py: run the training of 1 model 
- get_model.py: load one model into the code using our encoding convention 
- load_models.py: function to load several models consistently accross files


## Usage 
Several scripts can be runned and their usage is detailed below.

### Cross validation
The script ```crossval.py``` can be runned to perform the cross validation of all the different models we have presented. 
The first part of the script contains all the hyperparameters to fix and the parameters to use. This define the 'experiment' that will be run. The function `get_model` must be filled accordingly to the definitions of the parameters (i.e. using the same convention).
Once the parameters defined and the function completed, the rest of the script should be not changed. 
Further details on the parameters that can be changed are described in the documentation of the script. 

### Training
The script ```test.py``` gives an example on each of our 3 models presented above can be trained and evaluated. The parameters to be modified are specified in the documentation. It currently presents our best model.

### Evaluation
The script ```models_comparison_table.py``` provides model performance comparison across 25 epochs, and reports the mean and the standard deviation for each model error rates and training time.

### Plots
The script ```models_comparison_plots.py```allows plotting digit comparison and digit prediction test error rates for all the models we presented in the report. No modification is needed here.

 
