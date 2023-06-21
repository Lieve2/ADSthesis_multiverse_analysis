__doc__ = """this code is adapted from the code made by Nura Kawa, based on the original class map 
described by Raymaekers et al. (2021), which is available on CRAN as the R package 'classmap'.
The below code is an extension of the original class map and has a python implementation."""

# load libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.metrics import matthews_corrcoef
from scipy.stats import norm


def compPAC(model, X, y):
    """
            :param model: sklearn model, with probabilities, fitted to training data
            :param X:     (test) data for prediction (df)
            :param y:     true categories in target feature (array)
            :return:      Probability of Alternative Classification (PAC) from the trained classifier
    """

    # parameters
    n = X.shape[0]  # number of data points in X
    PAC = np.array([0.0]*n)  # initialize PAC array
    nlab = len(np.unique(y))  # number of classes

    # get fitted model probabilities
    model_probs = model.predict_proba(X)

    # case: two classes
    if nlab == 2:
        altint = 1 - y  # y will take values 0 or 1
        for i in range(n):
            PAC[i] = model_probs[i, altint[i]]
        return PAC

    # case: more than two classes
    ptrue = np.array([0.0]*n)  # array containing probability an item belongs to its true class
    palt = np.array([0.0]*n)  # array containing probability an item belongs to an alternative class
    for i in range(n):
        ptrue[i] = model_probs[i][y[i]]  # prob of the true class
        others = list(range(nlab))
        others.pop(y[i])  # indices of the other classes
        palt[i] = np.max(model_probs[i][others])  # most likely alternative class
        PAC[i] = (palt[i]) / (palt[i] + ptrue[i])  # PAC: conditional prob of alternative class

    return PAC


def compLocalFarness(X, y, k, metric='euclidean'):
    """
            :param X:       data for prediction, should be the same as what was used for PAC
            :param y:       corresponding labels of X
            :param k:       number of nearest neighbors to consider for localized farness computation
            :param metric:  distance metric for nearest neighbor search.
            :return:        localized farness computed from the data, independent of classifier
    """

    # find nearest neighbors with KD Tree
    kdt = KDTree(X, metric=metric)
    print('Searching for ' + str(k) + ' nearest neighbors for ' + str(X.shape[0]) +
          ' points. This could take some time!')
    dist, ind = kdt.query(X, k=k)  # get the nearest neighbor distances and indices
    print('Nearest neighbor search complete !')

    # array of epsilon_i (widths of epanechnikov kernels)
    epislon_arr = [dist[i][(k-1)] for i in range(len(dist))]

    # epanechnikov kernel weighting function
    ep_kernel = lambda x: (3/4)*(1 - (x*x))*(int(abs(x) <= 1))
    kernel_wt = lambda ep, d: (1/ep) * ep_kernel(x=(d/ep))

    # compute localized farness
    n = X.shape[0]  # number of rows in the data
    local_farness = np.array([0.0]*n)  # initialize local farness

    for i in range(n):
        local_dists = dist[i]  # distances from point i to its neighbors
        wts = [kernel_wt(ep=epislon_arr[i], d=local_dists[ii]) for ii in range(len(local_dists))]
        wts = wts / sum(wts)  # weight the local distances. wts should sum to 1.
        class_prob = sum(wts[y[ind[i]] == y[i]])  # Pr(i \in g_i)
        local_farness[i] = 1.0 - class_prob  # LF(i) = 1 - Pr(i \in g_i)

    # round to 4 decimal places, for simplicity
    # NOTE: np.abs() is used because sometimes we get -0.0. LF is always positive
    local_farness = np.abs(np.round(local_farness, 4))
    return local_farness


def plotExplanations(model, X, y, cl, k=10):
    """
            :param model:   fitted sklearn model
            :param X:       data for the model to make predictions
            :param y:       corresponding labels to X
            :param cl:      class, must be one of the classes in y
            :param k:       parameter for localized farness. Number of nearest neighbors
            :return:        data of localized class map of model X for elements of class cl in data X (df)
    """

    # to rescale LF for plot, we use qfunc,
    # quantile function of N(0,1) restricted to [0,a]
    qfunc = lambda x: abs(norm.ppf(x*(norm.pdf(4) - 0.5) + 0.5))

    # predictions from the model. We color the points by their predicted
    # class
    model_preds = model.predict(X)
    # performance
    model_perf = np.round(matthews_corrcoef(y_true=y, y_pred=model_preds), 4)
    class_perf = np.round(matthews_corrcoef(y_true=y[y == cl],
                                            y_pred=model_preds[y == cl]), 4)

    # compute PAC and LF
    PAC = compPAC(model, X, y)
    LF = compLocalFarness(X, y, k, metric='euclidean')

    # select the PAC of elements in specified class
    PAC_cl = PAC[y == cl]

    # rescale LF for view, select it for elements in specified class
    aLF_cl = qfunc(LF[y == cl])

    # get colors
    # for now using Tableau colors palette, which is limited to 10 colors
    nlab = len(np.unique(y))
    palette = ["#fd7e14", "#446e9b"]
    colors = np.array([palette[i] for i in model_preds[y == cl]])

    data = {'class': cl,
            'prob alternative': PAC_cl,
            'farness': aLF_cl,
            'colors': colors,
            'model perf': model_perf,
            'class perf': class_perf,
            'n labels': nlab}

    df_results = pd.DataFrame(data)

    return df_results
