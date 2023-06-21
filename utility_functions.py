from collections import defaultdict
from random import random

import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder, TargetEncoder
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.cluster import hierarchy
from scipy.cluster._optimal_leaf_ordering import squareform
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")


def random_imputation(df):
    """
            :param df:      a data frame with missing values
            :return:        complete data frame, where missing values are imputed by random imputation
    """

    df_imp = df.copy()
    for c in df_imp.columns:
        data = df_imp[c]
        mask = data.isnull()
        imputations = random.choices(data[~mask].values, k=mask.sum())
        data[mask] = imputations

        return df_imp


def permutate_features(X, threshold):
    """
            :param X:           the observed data (df)
            :param threshold:   the (cut) level for clustering
            :return:            X_new (the subset of observed data),
                                cluster_id_to_feat_id (list with cluster info)
    """

    # calculate correlation
    corr = spearmanr(X).correlation

    # ensure symmetry
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # distance matrix and linkage with Ward's
    dist_matrix = 1 - np.abs(corr)
    dist_link = hierarchy.ward(squareform(dist_matrix))

    # group features in clusters and keep one feature per cluster
    cluster_ids = hierarchy.fcluster(dist_link, threshold, criterion='distance')
    cluster_id_to_feat_id = defaultdict(list)

    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feat_id[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feat_id.values()]

    X_new = X.iloc[:, selected_features]

    return X_new, cluster_id_to_feat_id


def encode_scale_data_perm(data, tuning_target, threshold, num_feat):
    """
            :param data:            the complete data set
            :param tuning_target:   the target feature
            :param threshold:       the (cut) level for clustering
            :param num_feat:        a list with numeric features
            :return:                X_new (the complete and scaled/encoded subset of observed data)
                                    y (the binarized target feature),
                                    features (list with features present in X_new),
                                    clusters (list with cluster information)
    """

    # encode objects in data
    enc = OrdinalEncoder()
    data_obj = data[data.columns.intersection(num_feat)]
    enc.fit(data_obj)
    encoding = enc.fit_transform(data[data_obj.columns])

    c = 0

    for i in data_obj.columns:
        data[i] = encoding[:, c]
        c += 1

    # binarize target to 0 (missing) and 1 (non-missing)
    y = data[tuning_target].notnull().astype('int')

    # drop target from observed data
    X = data.drop(tuning_target, axis=1)

    if tuning_target in num_feat:
        num_feat = [i for i in num_feat if i != tuning_target]

    cat_feat = X.drop(num_feat, axis=1).columns

    for c in cat_feat:
        # missing values as new category in the categorical data
        X[c] = X[c].fillna(-1).astype('category', copy=False)
    for n in num_feat:
        X[n] = random_imputation(X[n].to_frame())

    # define scaling-encoding pipeline
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", TargetEncoder())])
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, num_feat),
                                                   ("cat", categorical_transformer, cat_feat)])

    # extract feature names
    features_compl = X.columns

    # scale/encode the observed data
    X_scaled = pd.DataFrame(preprocessor.fit_transform(X, y), columns=features_compl)

    # remove multicollinearity
    X_new, clusters = permutate_features(X_scaled, threshold)
    features = X_new.columns

    return X_new, y, features, clusters


def rosenbrock(vector, a=1, b=100):
    """
    :param vector:  a vector to calculate the rosenbrock function on
    :param a:       variable in the function, default a=1
    :param b:       variable in the function, default b=100
    :return solution of the rosenbrock function for the given vector

    f(x, y) = (a-x)^2 + b(y-x^2)^2
    """

    vector = np.array(vector)

    return (a - vector[0])**2 + b * (vector[1] - vector[0]**2)**2


def rastrigin(vector):
    """
            :param vector:  a vector to calculate the rastrigin function on
            :return solution of the rastrigin function for the given vector

            f(x) = 10*n + Sigma { x_i^2 - 10*cos(2*PI*x_i) }

    """

    vector = np.array(vector)

    return 10 * vector.size + sum(vector * vector - 10 * np.cos(2 * np.pi * vector))


def multi_csv_to_df(files, axis=0, index_col=None):
    """
            :param files:       list of csv file paths
            :param axis:        on what axis to aggregate the files (rows (0) or columns (1))
            :param index_col:   index column to use, if applicable
            :return:            a single data frame of the aggregated csv files
    """

    lst = []

    # files to alphabetical order
    files_sorted = sorted(files)

    for filename in files_sorted:
        df = pd.read_csv(filename, index_col=index_col, header=0)
        lst.append(df)

    df_results = pd.concat(lst, axis=axis, ignore_index=True)
    return df_results


def ConvergencePlot(cost):
    """
    Monitors convergence.
    Parameters:
    ----------
        :param dict cost: mean and best cost over cycles/generations as returned
                          by an optimiser.
    """

    font = FontProperties()
    font.set_size('larger')
    labels = ["Best Cost Function", "Mean Cost Function"]
    plt.figure(figsize=(12.5, 4))
    plt.plot(range(len(cost["best"])), cost["best"], label=labels[0])
    plt.scatter(range(len(cost["mean"])), cost["mean"], color='red', label=labels[1])
    plt.xlabel("Iteration #")
    plt.ylabel("Value [-]")
    plt.legend(loc="best", prop=font)
    plt.xlim([0, len(cost["mean"])])
    plt.grid()
    plt.show()
