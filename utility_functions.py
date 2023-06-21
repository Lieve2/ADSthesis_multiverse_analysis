from collections import defaultdict
from random import random

import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder, TargetEncoder
from scipy.cluster import hierarchy
from scipy.cluster._optimal_leaf_ordering import squareform
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# voor beschrijving functions:
# """
#         :param files: list of csv file paths
#         :param axis:  on what axis to aggregate the files (rows (0) or columns (1))
#         :return:      a single data frame of the aggregated csv files
#         """

def random_imputation(df):

    df_imp = df.copy()
    for c in df_imp.columns:
        data = df_imp[c]
        mask = data.isnull()
        imputations = random.choices(data[~mask].values, k = mask.sum())
        data[mask] = imputations

        return df_imp

def permutate_features(X, threshold):

    corr = spearmanr(X).correlation

    # ensure symmetry
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)


    # distance matrix and linkage with Ward's
    dist_matrix = 1 - np.abs(corr)
    dist_link = hierarchy.ward(squareform(dist_matrix))

    # group features in clusters and keep one feature per cluster
    cluster_ids =hierarchy.fcluster(dist_link, threshold, criterion='distance')
    cluster_id_to_feat_id = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feat_id[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feat_id.values()]

    X_new = X.iloc[:, selected_features]

    return X_new, cluster_id_to_feat_id

def encode_scale_data_perm(data, tuning_target, threshold, num_feat):

    # encode objects in data
    enc = OrdinalEncoder()
    #data_obj = data.select_dtypes(include=object)
    data_obj = data[data.columns.intersection(num_feat)]
    enc.fit(data_obj)
    encoding = enc.fit_transform(data[data_obj.columns])

    c = 0

    for i in data_obj.columns:
        data[i] = encoding[:, c]
        c += 1

    y = data[tuning_target].notnull().astype('int')
    X = data

    if tuning_target in num_feat:
        num_feat = [i for i in num_feat if i != tuning_target]

    cat_feat = X.drop(num_feat, axis=1).columns

    for c in cat_feat:
        X[c] = X[c].fillna(-1).astype('category', copy=False)
    for n in num_feat:
        X[n] = random_imputation(X[n].to_frame())

    # scaling-encoding pipeline
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", TargetEncoder())])
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, num_feat),
                                                   ("cat", categorical_transformer, cat_feat)])

    features_compl = X.columns

    X_scaled = pd.DataFrame(preprocessor.fit_transform(X,y), columns=features_compl)

    # remove multicollinearity
    X_new, clusters = permutate_features(X_scaled, threshold)  # X_scaled
    features = X_new.columns

    return X_new, y, features, clusters

def encode_scale_data(data, num_feat, tuning_target):

    # encode objects in data
    enc = OrdinalEncoder()
    data_obj = data.select_dtypes(include=object)
    enc.fit(data_obj)
    encoding = enc.fit_transform(data[data_obj.columns])

    c = 0

    for i in data_obj.columns:
        data[i] = encoding[:, c]
        c += 1

    # set target for scaling
    y = data[tuning_target].notnull().astype('int')

    # label missing values
    X = data.fillna('missing')

    cat_feat = X.drop(num_feat, axis=1).columns

    # set categorical data to corresponding type
    for c in cat_feat:
        #X[c] = X[c].fillna(-1).astype('category', copy=False)  # missing values as new category in the categorical data
        X[c] = X[c].astype('category', copy=False)

    # define scaling-encoding pipeline
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", TargetEncoder())])
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, num_feat),
                                                   ("cat", categorical_transformer, cat_feat)])

    # extract features
    features_compl = X.columns

    # perform scale-encode pipeline and put transformed data into df
    X_scaled = pd.DataFrame(preprocessor.fit_transform(X, y), columns=features_compl)

    # impute missing values back into data
    for f in features_compl:
        # add missingness back into df
        X_scaled.loc[X[f] == 'missing', f] = np.nan

    return X_scaled

def rosenbrock(vector, a=1, b=100):
    """f(x, y) = (a-x)^2 + b(y-x^2)^2"""

    vector = np.array(vector)

    return (a - vector[0])**2 + b * (vector[1] - vector[0]**2)**2


def rastrigin(vector):
    """                     n
            f(x) = 10*n + Sigma { x_i^2 - 10*cos(2*PI*x_i) }
                           i=1
                           with xi within [-5.12:5.12]

    """

    vector = np.array(vector)

    return 10 * vector.size + sum(vector * vector - 10 * np.cos(2 * np.pi * vector))

