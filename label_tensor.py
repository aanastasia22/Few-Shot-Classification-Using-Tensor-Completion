import numpy as np
from statistics import mode

def label_tensor(inputs,targets,I):
    """
    Create the label tensor X with indices in the inputs and elements in the targets of the training samples
    Inputs:
            inputs  : training samples (n_samples x N array)
            targets : training labels (n_samples x 1 vector)
            I       : size of the tensor X in each dimension (N x 1 vector)
    Output:
            X       : I(1) x I(2) x ... x I(N) tensor with indices in the inputs
                      and elements in the targets
    """
    
    print('Create label tensor')
    X = np.zeros(I)
    # Find the unique sample values
    uniq_values = np.unique(inputs, axis=0).astype('uint16')
    for values in uniq_values:
        # Fill the label tensor in the position indicating by the sample values with the majority vote 
        # of the corresponding labels
        idx = np.where((inputs==values).all(axis=1))[0]
        X[tuple(values)] = mode(np.unique(targets[idx]))
    
    return X