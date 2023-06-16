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
    # corr = X.corr()
    #print(corr)

    # ensure symmetry
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)


    # distance matrix and linkage with Ward's
    dist_matrix = 1 - np.abs(corr)
    dist_link = hierarchy.ward(squareform(dist_matrix))
    # dist_matrix = 1 - corr.abs().values
    # dist_link = hierarchy.linkage(dist_matrix, method="ward")

    # group features in clusters and keep one feature per cluster
    cluster_ids =hierarchy.fcluster(dist_link, threshold, criterion='distance')
    cluster_id_to_feat_id = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feat_id[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feat_id.values()]

    # print(cluster_id_to_feat_id)

    X_new = X.iloc[:, selected_features]

    # print(X_new)

    return X_new, cluster_id_to_feat_id

def encode_scale_data_perm(data, tuning_target, threshold, num_feat):
    # encode objects in data
    enc = OrdinalEncoder()
    data_obj = data.select_dtypes(include=object)
    enc.fit(data_obj)
    encoding = enc.fit_transform(data[data_obj.columns])

    c = 0

    for i in data_obj.columns:
        data[i] = encoding[:, c]
        c += 1

    y = data[tuning_target].notnull().astype('int')  # change to df.notnull().astype('int')

    # X = data.drop(tuning_target, axis=1)
    X = data

    # cat_feat = X.drop(['dweight', 'pspwght',
    #                        'pweight', 'anweight'], axis=1).columns

    # num_feat = ['dweight', 'pspwght','pweight', 'anweight']

    cat_feat = X.drop(num_feat, axis=1).columns


    # scaler = StandardScaler()

    for c in cat_feat:
        X[c] = X[c].fillna(-1).astype('category', copy=False) # missing values as new category in the categorical data
    # for n in num_feat:
    #     X[n] = scaler.fit_transform(X[n].to_frame())
    #print(features)

    #encoder = TargetEncoder()
    #X_scaled = encoder.fit_transform(X, y)

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", TargetEncoder())])
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, num_feat),
                                                   ("cat", categorical_transformer, cat_feat)])

    features_compl = X.columns

    X_scaled = pd.DataFrame(preprocessor.fit_transform(X,y), columns=features_compl)

    # remove multicollinearity
    X_new, clusters = permutate_features(X_scaled, threshold)  # X_scaled
    features = X_new.columns


    # scale/target encode the data
    # scaler = StandardScaler()
    # scaler.fit(X_new)
    # X_scaled = scaler.transform(X_new)
    # for n in num_feat:
    #     X[n] = scaler.fit_transform(X[n].to_frame())
    # encoder = TargetEncoder()
    # X_scaled = encoder.fit_transform(X_new, y)
    #print(X_scaled)

    #print(X_new, y)

    return X_new, y, features, clusters #X_scaled

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

        # X_scaled[f] = X_scaled[f].loc[X[f] == 'missing']
    # print(X_scaled['netustm'].loc[X['netustm'] == 'missing'].index)
    # X_scaled = pd.DataFrame(list(X_scaled)) #, columns=features_compl

    # print(X_scaled[X_scaled[['netustm', 'yrbrn3']].isnull().any(axis=1)])

    return X_scaled


# # load data
# data = pd.read_csv('ESS8 data/ESS8_subset_cleaned_timeadj_wmissingvals.csv', low_memory=False)
#
# #define numerical features
# num_feat = ['dweight', 'pspwght','pweight', 'anweight', 'inwtm']
#
# print(encode_scale_data(data, num_feat, 'netustm'))

# impute 'missing' with np.nan

