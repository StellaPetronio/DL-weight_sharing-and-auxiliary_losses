########################################################################################################################
# Parameters for the models
p_dropout, nb_channels, nb_lay_1, dim_lin1, nb_lay_2, dim_lin2, nb_lay_latent, dim_latent, kernel_sz = 0.1, 32, 4, 25, 4, 25, 4, 2, 3

#parameters best model
p_dropout_best, nb_channels_best, nb_lay_1_best, dim_lin1_best, nb_lay_2_best, dim_lin2_best, nb_lay_latent_best, dim_latent_best = 0.5, 32, 3, 25, 3, 25, 3, 8


###################################################ù#####################################################################

model_comparisons = [ 
    ["A ", "g-", # No aux. loss & no weight sharing 
    (1, 1, p_dropout, nb_channels, nb_lay_1, dim_lin1, nb_lay_2, dim_lin2, None, None), 1],
                     
    ["B ", "b-",#Aux. loss & no weight sharing
    (1, 0.5, p_dropout, nb_channels, nb_lay_1, dim_lin1, nb_lay_2, dim_lin2, None, None), 0.5],
                     
    ["C ", "r-",#No aux. loss & weight sharing (2 convs)
    (3, 1, p_dropout, nb_channels, nb_lay_1, dim_lin1, nb_lay_2, dim_lin2, nb_lay_latent, dim_latent), 1],
                      
    ["D ", "y-",#No aux. loss & weight sharing (3 convs)
    (2, 1, p_dropout, nb_channels, nb_lay_1, dim_lin1, nb_lay_2, dim_lin2, nb_lay_latent, dim_latent), 1],
                      
    ["E ", "c-",#Aux. loss & weigth sharing (3 convs)
    (2, 0.5, p_dropout, nb_channels, nb_lay_1, dim_lin1, nb_lay_2, dim_lin2, nb_lay_latent, 2), 0.5],
                     
    ["F ", "k-",#Best Model
    (2, 0.5, p_dropout_best, nb_channels_best, nb_lay_1_best, dim_lin1_best, nb_lay_2_best, dim_lin2_best, nb_lay_latent_best, dim_latent_best), 0.5]
    ]

def load_models():
    return model_comparisons
