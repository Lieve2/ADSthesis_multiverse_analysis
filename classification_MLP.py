import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from category_encoders import TargetEncoder
from pipeline_MLP import KfoldCV


def random_imputation(df):

    df_imp = df.copy()
    for c in df_imp.columns:
        data = df_imp[c]
        mask = data.isnull()
        imputations = random.choices(data[~mask].values, k = mask.sum())
        data[mask] = imputations

        return df_imp

def permutate_features(X, threshold):

    # calculate Spearman correlation
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

    # impute missing values in observed data
    for c in cat_feat:
        X[c] = X[c].fillna(-1).astype('category', copy=False) # missing values as new category in the categorical data
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
    X_scaled = pd.DataFrame(preprocessor.fit_transform(X,y), columns=features_compl)

    # remove multicollinearity
    X_new, clusters = permutate_features(X_scaled, threshold)  # X_scaled
    features = X_new.columns

    return X_new, y, features, clusters

def hyperparameter_tuning(data, targets, threshold, filename, num_feat, PCA_transformed=False):

    for target in targets:

        # extract pre-processsed data, subset of features used, and cluster info
        X, y, features, clusters = encode_scale_data_perm(data, target, threshold, num_feat)

        # store cluster info in csv file
        df_clusters = pd.DataFrame.from_dict(clusters, orient='index')
        df_clusters.to_csv('clusters info/MLP/{}_{}.csv'.format(filename, target))

        # timer - start
        t = time.process_time()

        # perform k-fold CV
        KfoldCV(3, X, y,
                filename='{}_{}'.format(filename, target),
                features=features)

        # timer - end
        elapsed_time = time.process_time() - t
        print(f"Elapsed time for K-fold CV {target}: {elapsed_time} sec")


## ------- end of functions --------- ##

# load data
data = pd.read_csv('ESS8 data/ESS8_subset_cleaned_timeadj_wmissingvals.csv', low_memory=False)

# make df with missingness percentage of features with missingness
missing_cols = data.columns[data.isnull().any()].to_list()
percent_missing = data[missing_cols].isnull().sum() * 100 / len(data)
missing_info = pd.DataFrame({'column name':missing_cols,
                             'percentage missing':percent_missing})

# extract rows with > 5% missing (75 features) and set as targets
many_missing = missing_info[missing_info['percentage missing'] > 5]
targets = many_missing['column name'].tolist()

# define numerical features
num_feat = ['dweight', 'pspwght','pweight', 'anweight', 'nwspol',
            'netustm','agea', 'eduyrs', 'wkhct', 'wkhtot',
            'wkhtotp', 'inwtm']

# start the modeling pipeline
hyperparameter_tuning(data=data, targets=targets, num_feat=num_feat, threshold=1, filename='MLP')









