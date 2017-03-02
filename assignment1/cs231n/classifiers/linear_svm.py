import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops)
    Inputs:
    - W: C x D array of weights (C=classes, D=dimension)
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[0]
    num_train = X.shape[1]
    loss = 0.0
    for image in range(num_train):
        scores = W.dot(X[:, image])
        correct_class = y[image]
        correct_class_score = scores[correct_class]

        for output_class in range(num_classes):
            if output_class != correct_class:
                margin = scores[output_class] - correct_class_score + 1
                if margin > 0:
                    loss += margin
                    dW[output_class, :] += X[:, image].T
                    dW[correct_class, :] -= X[:, image].T


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Same for dW
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    # Same for dW
    dW += reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    num_train = X.shape[1]
    scores = W.dot(X).T

    correct_class_indexes = (range(scores.shape[0]), y)
    correct_class_scores = scores[correct_class_indexes]

    # margin = class score - correct class score + 1
    margins = scores - correct_class_scores.reshape(-1, 1) + 1
    
    #margins for correct classes are 0
    margins[correct_class_indexes] = 0

    #substitute margins smaller then 0 with 0,
    #this is the same of performing max(0,margin)
    margins = np.where(margins>0, margins, 0)

    

    #calculate the loss    
    losses = np.sum(margins, axis=0) #1 loss for every image
    loss = np.sum(losses)

    #print(losses.shape)
    #print(loss)

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    # set margins for uncorrect classes to 1
    margins[margins>0] = 1 

    # set margins for correct classes to -1
    margins[correct_class_indexes] = margins.sum(axis=1) * -1

    #weight the margin with X
    dW = X.dot(margins).T
    
    #############################################################################
    #                         END OF YOUR CODE                                  #
    #############################################################################


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    # Same for dW.
    loss /= num_train
    dW /= num_train


    # Add regularization to the loss and dW
    loss += 0.5 * reg * np.sum(W * W)
    dW = dW + reg * W

    return loss, dW
