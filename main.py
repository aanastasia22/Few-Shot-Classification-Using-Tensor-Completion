import numpy as np
import random
from label_tensor import *
from label_tensor_completion import *
from compute_accuracy import *
import math
import scipy.io as sio

random.seed(10)

# Parameters
p_train = 0.8 # Percentage of training samples
p_val = 0.3   # Percentage of training samples used for validation
lam = 0.5     # Lambda parameter
iter = 100   # Number of maximum iterations
tol = 1e-10   # Tolerance for stopping criterion
n_sim = 10    # Number of simulations
delta = 0.1     # Delta parameter
step = 0.01    # Step size of the gradient step for score tensor update
rank = []     # Possible rank values (as percentage of the original dimensions) for cross-validation
R = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
for i in range(len(R)):
    rank.append(np.array([R[i],R[i],R[i],R[i],R[i],R[i],R[i],R[i],R[i],1]))

# Load the data
data_targets = sio.loadmat('./Data/breast_cancer.mat')
dim_label_tensor = np.array([9,3,12,13,2,3,2,5,2]) # Dimensions of the label tensor
data = data_targets['data']-1
num = data.shape[0]
num_train = round(num*p_train)
num_test = num-num_train
targets = data_targets['targets']
del(data_targets)

# Initialization
acc_te = np.zeros(n_sim)
acc_tr = np.zeros(n_sim)
acc_val = np.zeros(n_sim)
loss = []
acc_val_train = []
rank_best = []
# Run n_sim Monte-Carlo Simulations
for i in range(n_sim):
    print('Number of simulation: {}'.format(i+1))
    # Training and Testing Data
    num_classes = len(np.unique(targets))
    idx_test = random.sample(range(num),num_test)
    idx = list(set(range(num)) - set(idx_test))
    idx_val = random.sample(idx,math.floor(p_val*num_train))
    idx_train = list(set(idx) - set(idx_val))
    train = data[idx_train]
    targets_train = np.squeeze(targets[idx_train])
    val = data[idx_val]
    targets_val = np.squeeze(targets[idx_val])
    test = data[idx_test]
    targets_test = np.squeeze(targets[idx_test])

    # Create the label uncertainty tensor
    N = train.shape[1]
    X = label_tensor(train,targets_train,dim_label_tensor)

    # Perform cross-validation to find the best rank value
    acc_best = 0
    rank_best.append(rank[0])
    acc_val_train.append([])
    loss.append([])
    for j in range(len(rank)):
        # Label Tensor Completion
        Xpred, sum_loss, acc_val_tr = label_tensor_completion_hinge_loss_gd(X,lam,rank[j],delta,iter,tol,step,val,targets_val)
        acc_val_train[i].append(acc_val_tr)
        loss[i] = sum_loss
        # Find the best rank according to validation accuracy
        if np.max(acc_val_tr)>=acc_best:
            acc_best = np.max(acc_val_tr)
            Xpred_best = Xpred.copy()
            rank_best[i] = rank[j].copy()

    # Validation accuracy
    pred_val, acc_val[i] = compute_accuracy(val,targets_val,Xpred_best)
    print('Validation Accuracy: {:.3}'.format(acc_val[i]))

    # Label Tensor Completion using the best rank
    train = np.concatenate((train, val), axis=0)
    targets_train = np.concatenate((targets_train,targets_val), axis=None)
    X = label_tensor(train,targets_train,dim_label_tensor)
    Xpred_best, sum_loss, _ = label_tensor_completion_hinge_loss_gd(X,lam,rank_best[i],delta,iter,tol,step)

    # Training accuracy
    pred_train, acc_tr[i] = compute_accuracy(train,targets_train,Xpred_best)
    print('Training Accuracy: {:.3}'.format(acc_tr[i]))

    # Testing accuracy
    pred_test, acc_te[i] = compute_accuracy(test,targets_test,Xpred_best)
    print('Testing Accuracy: {:.3}'.format(acc_te[i]))

print('Mean Testing Accuracy {} and standard deviation {}'.format(np.mean(acc_te),np.std(acc_te)))
print('Mean Validation Accuracy {} and standard deviation {}'.format(np.mean(acc_val),np.std(acc_val)))
print('Mean Training Accuracy {} and standard deviation {}'.format(np.mean(acc_tr),np.std(acc_tr)))

results = {'RR':R,'X':X,'RR_best':rank_best,'acc_val_train':acc_val_train,'acc_val':acc_val,
    'acc_tr':acc_tr,'acc_te':acc_te,'loss':loss}
sio.savemat('C:/Users/aidini/Documents/Tensor_Learning_for_Classification/results_paper_new/breast_cancer_train229_test57_lam05_100iter_1010tol_hingeloss_gd1000_delta01_pval03_10nsim_step001.mat',results)