
import math
import random
import time
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import parallel_backend
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from category_encoders import TargetEncoder

from pipeline_SVM import KfoldCV #!!!!!!

import scipy.spatial.distance as ssd


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
    # data_obj = data.select_dtypes(include=object)
    data_obj = data[data.columns.intersection(num_feat)]
    enc.fit(data_obj)
    encoding = enc.fit_transform(data[data_obj.columns])

    c = 0

    for i in data_obj.columns:
        data[i] = encoding[:, c]
        c += 1

    y = data[tuning_target].notnull().astype('int')  # change to df.notnull().astype('int')

    #X = data.drop(tuning_target, axis=1).fillna(-1) #was -999
    #X_imp = random_imputation(X) #random imputation (per feature)
    X = data.drop(tuning_target, axis=1)

    # cat_feat = X.drop(['dweight', 'pspwght',
    #                        'pweight', 'anweight'], axis=1).columns

    # num_feat = ['dweight', 'pspwght','pweight', 'anweight']

    if tuning_target in num_feat:
        # print(num_feat.remove(str(tuning_target)))
        num_feat = [i for i in num_feat if i != tuning_target]

    cat_feat = X.drop(num_feat, axis=1).columns


    # scaler = StandardScaler()

    for c in cat_feat:
        X[c] = X[c].fillna(-1).astype('category', copy=False) # missing values as new category in the categorical data
    for n in num_feat:
        X[n] = random_imputation(X[n].to_frame())


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

def hyperparameter_tuning(data, targets, threshold, filename, num_feat, PCA_transformed=False):

    for target in targets:

        X, y, features, clusters = encode_scale_data_perm(data, target, threshold, num_feat)
        # print(features)

        df_clusters = pd.DataFrame.from_dict(clusters, orient='index')

        df_clusters.to_csv('clusters info/SVM/{}_{}.csv'.format(filename, target))

        # timer - start
        t = time.process_time()

        # perform k-fold CV
        KfoldCV(3, X, y,
                filename='{}_{}'.format(filename, target),
                features=features)

        # timer - end
        elapsed_time = time.process_time() - t
        print(f"Elapsed time for K-fold CV {target}: {elapsed_time} sec")

        # extract_feature_importance(data=data,
        #                            tuning_target=tuning_target,
        #                            pca_post=pca_post,
        #                            filename=filename)








## ------- end of functions --------- ##

# load data
data = pd.read_csv('ESS8 data/ESS8_cleaned_wmissingvals.csv', low_memory=False)

# load cleaned data
#data2 = pd.read_csv('ESS8 data/ESS8_subset_cleaned_wmissingvals.csv', low_memory=False)
data3 = pd.read_csv('ESS8 data/ESS8_subset_cleaned_timeadj_wmissingvals.csv', low_memory=False)
# make df with missingness percentage of features with missingness
missing_cols = data3.columns[data3.isnull().any()].to_list()
percent_missing = data3[missing_cols].isnull().sum() * 100 / len(data3)
missing_info = pd.DataFrame({'column name':missing_cols,
                             'percentage missing':percent_missing})

# extract rows with > 5% missing (75 features)
many_missing = missing_info[missing_info['percentage missing'] > 5]
# print(len(many_missing))

targets3 = many_missing['column name'].tolist()
num_feat = ['dweight', 'pspwght','pweight', 'anweight', 'nwspol',
            'netustm','agea', 'eduyrs', 'wkhct', 'wkhtot',
            'wkhtotp', 'inwtm']

hyperparameter_tuning(data=data3, targets=targets3, num_feat=num_feat, threshold=1, filename='SVM')









