from framework import *

def get_model(params, device):
    """
    Given a list of parameters as a tuple, return the model encoded by those hyper-parameters

    Parameter definition 
    --------------------
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
    """
    index, w1, p, nb_channels, n1, dim1, n2, dim2, nl, diml = params
    use_sharing = index > 1
    if not use_sharing: 
        model = ModelNoWeightSharing(n_channels_conv=nb_channels, 
                                dim_linear1=dim1,
                                n_layers_linear1=n1, 
                                dim_linear2=dim2,
                                n_layers_linear2=n2,
                                p_dropout = p,
                                )
    else:
        if index <= 3 :
            use_long = index == 2
            model = ModelWeightSharing(use_long=use_long,
                                      nb_channels=nb_channels,
                                      dim_linear1=dim1,
                                      n_layers_linear1=n1,
                                      dim_linear2=dim2,
                                      n_layers_linear2=n2,
                                      n_layers_latent=nl,
                                      dim_latent=diml,
                                      p=p,
                                      )
        else:
            model = ModelWeightSharingDeep(n_channels_conv=nb_channels,
                                dim_linear1=dim1,
                                n_layers_linear1=n1,
                                dim_linear2=dim2,
                                n_layers_linear2=n2,
                                p_dropout = p,
                                )
    model.to(device)
    return model
