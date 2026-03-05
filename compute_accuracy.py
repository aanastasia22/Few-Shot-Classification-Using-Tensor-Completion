import numpy as np 

def compute_accuracy(samples,targets,X):
    """ 
    Find the predicted samples and compute accuracy (%)
    Inputs: 
            samples : n_samples x N array
            targets : true labels (n_samples x 1 vector)
            X       : label tensor of size I(1) x I(2) x ... x I(N) 
    Outputs:
            pred    : predicted labels of the samples (n_samples x 1 vector) 
            acc     : accuracy (%)
    """

    # Find the predicted labels by taking the value of X in the position indicating by the samples
    pred = []
    for s in samples:
        pred.append(X[tuple(s)])
    pred = np.array(pred)
    # Compute the accuracy comparing the true labels with the predicted ones
    try:
        num = len(targets)
    except:
        num = 1
    idx = np.where(targets == pred)
    acc = (100*len(idx[0]))/num

    return pred, acc