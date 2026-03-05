import numpy as np
import tensorly as tl
from compute_accuracy import *

np.random.seed(10)

def grad_hinge_loss(score,true_labels,delta):
    """
    Compute the gradient of hinge loss for each class and each sample
    Inputs:
            score       : score values (num_classes x num_samples array)
            true_labels : true labels of the samples (num_samples x 1 vector)
            delta       : margin for the loss (scalar)
    Output:
            grad : gradient of hinge loss for each class and each sample (num_classes x num_samples array)
    """

    grad = np.zeros(score.shape)
    idx = np.nonzero(true_labels)
    for i in range(len(idx[0])):
        s = 0
        for j in range(score.shape[0]):
            if j != (true_labels[idx[0][i]]-1):
                v = score[j,idx[0][i]]-score[true_labels[idx[0][i]]-1,idx[0][i]]+delta
                if v > 0.0:
                    grad[j,idx[0][i]] = -1
                    s += 1
        grad[true_labels[idx[0][i]]-1,idx[0][i]] = s
    return -grad

def compute_hinge_loss(score,true_labels,delta):
    """
    Compute the hinge loss for each sample
    Inputs:
            score       : score values (num_classes x num_samples array)
            true_labels : true labels of the samples (num_samples x 1 vector)
            delta       : margin for the loss (scalar)
    Output:
            loss : hinge loss for each sample (num_samples x 1 vector)
    """

    loss = np.zeros(true_labels.shape)
    idx = np.nonzero(true_labels)
    for i in range(len(idx[0])):
        for j in range(score.shape[0]):
            if j != (true_labels[idx[0][i]]-1):
                v = score[j,idx[0][i]]-score[true_labels[idx[0][i]]-1,idx[0][i]]+delta
                if v > 0.0:
                    loss[idx[0][i]] += v
    return loss

def label_tensor_completion_hinge_loss_gd(X,lam,rank,delta,iter,tol,step,val=[],targets_val=[]):
    """
    Label tensor completion
    Inputs:
            X           : label tensor
            lam         : lambda parameter (scalar)
            rank        : multilinear rank of the label tensor as percentage of the original dimensions
                          ((N+1) x 1 vector)
            delta       : delta parameter (scalar)
            iter        : number of maximum iterations (scalar)
            tol         : tolerance for stopping criterion (scalar)
            step        : step size of the gradient step for score tensor update (scalar)
            val         : validation samples (n_samples x N array)
            targets_val : true labels of validation samples (n_samples x 1 vector)
    Outputs:
            X_best   : recovered label tensor
            sum_loss : loss at each iteration (list)
            acc_val  : validation accuracy at each iteration (list)
    """
    X = X.astype('uint8')

    # Find the available observations
    idx = np.nonzero(X!=0)
    perc = (len(idx[0])*100)/np.prod(X.shape)
    print('Percentage of available measurements: {:.2} %'.format(perc))
    
    # Size of the score tensor
    num_classes = len(np.unique(X[idx]))
    N = X.ndim
    dim = X.shape
    dim = np.concatenate((dim,np.array([num_classes])))

    # Initialization
    A = np.random.rand(*dim) # Score tensor
    R = list(np.round(rank*dim).astype('uint16')) # Multilinear rank of the score tensor
    D = [] # Factor matrices
    for n in range(N+1):
        if R[n] == 0:
            R[n] = 1
        D.append(np.random.rand(dim[n],R[n]))
        Q, r = np.linalg.qr(D[n]) # Impose orthogonal contraint
        D[n] = Q
    print('Rank {}'.format(R[:N]))
    G = tl.tenalg.multi_mode_dot(A,D,transpose=True) # Core tensor
    Z = tl.tenalg.multi_mode_dot(G,D) # Tucker composition corresponding to the score tensor
    acc = []
    sum_loss = []
    acc_val = []
    best_acc = -1
    cnt_acc = 0
    thresh = 5
    print('Label Tensor Completion')
    for i in range(iter):
        # Update the factor matrices
        D_init = D
        for n in range(N+1):
            CN = tl.tenalg.multi_mode_dot(G,D_init,skip=n)
            Cnn = tl.unfold(CN,n)
            D[n] = tl.unfold(A,n)@np.linalg.pinv(Cnn)
            # Impose orthogonal contraint
            Q,r = np.linalg.qr(D[n])
            D[n] = Q
        # Update the core tensor G
        G = tl.tenalg.multi_mode_dot(A,D,transpose=True)
        # Update the score tensor A
        Z = tl.tenalg.multi_mode_dot(G,D)
        for j in range(1000):
            B = grad_hinge_loss(tl.unfold(A,N),X.flatten(),delta)
            B = tl.fold(B,N,dim)
            A = A-step*(B+lam*(A-Z))
        # Compute accuracy and NRMSE
        Xpred = np.argmax(A,axis=N)+1
        idx_eq = np.where(X[idx] == Xpred[idx])
        acc.append((100*len(idx_eq[0]))/len(idx[0]))
        # Compute Loss
        hinge_loss = sum(compute_hinge_loss(tl.unfold(A,N),X.flatten(),delta))/len(idx[0])
        sum_loss.append(hinge_loss+(lam/2)*(np.linalg.norm(Z.flatten()-A.flatten())/np.linalg.norm(A.flatten())))
        # print('Iteration: {}, Accuracy {:.3}, Loss {:.5}'.format(i+1,acc[i],sum_loss[i]))
        # Stopping criterion
        if len(val)!=0: # Stop according to validation accuracy
            _, acv = compute_accuracy(val,targets_val,Xpred)
            acc_val.append(acv)
            # print('Validation Accuracy: {:.3}'.format(acc_val[i]))
            if acc_val[i] >= best_acc:
                best_acc = acc_val[i]
                X_best = Xpred.copy()
            if (i!=0)and(acc_val[i]<acc_val[i-1]):
                cnt_acc += 1
            else:
                cnt_acc = 0
            if ((i!=0)and(abs(sum_loss[i-1]-sum_loss[i])<tol))or(cnt_acc >= thresh)or(sum_loss[i]<tol):
                break
        else:
            X_best = Xpred.copy()
            if (i!=0)and(sum_loss[i-1]<sum_loss[i]):
                X_best = Xprev.copy()
                cnt_acc += 1
            else:
                cnt_acc = 0
            if ((i!=0)and(abs(sum_loss[i-1]-sum_loss[i])<tol))or(cnt_acc >= thresh)or(sum_loss[i]<tol):
                break
            else:
                Xprev = X_best.copy()

    return X_best, sum_loss, acc_val