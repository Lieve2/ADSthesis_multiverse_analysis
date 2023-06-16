from collections import defaultdict
import random

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from matplotlib import pyplot as plt


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
    data_obj = data[data.columns.intersection(num_feat)]
    enc.fit(data_obj)
    encoding = enc.fit_transform(data[data_obj.columns])

    c = 0

    for i in data_obj.columns:
        data[i] = encoding[:, c]
        c += 1

    y = data[tuning_target].notnull().astype('int')
    X = data.drop(tuning_target, axis=1)

    if tuning_target in num_feat:
        num_feat = [i for i in num_feat if i != tuning_target]

    cat_feat = X.drop(num_feat, axis=1).columns

    for c in cat_feat:
        X[c] = X[c].fillna(-1).astype('category', copy=False) # missing values as new category in the categorical data
    for n in num_feat:
        X[n] = random_imputation(X[n].to_frame())

# scale-encoding pipeline
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", TargetEncoder())])
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, num_feat),
                                                   ("cat", categorical_transformer, cat_feat)])

    features_compl = X.columns

    X_scaled = pd.DataFrame(preprocessor.fit_transform(X,y), columns=features_compl)

    # remove multicollinearity
    X_new, clusters = permutate_features(X_scaled, threshold)
    features = X_new.columns

    return X_new, y, features, clusters

def epoch_test(data, targets, threshold, num_feat, classifier):

    for target in targets:

        X, y, features, clusters = encode_scale_data_perm(data, target, threshold, num_feat)

        X_train, X_hold, y_train, y_hold = train_test_split(X, y, train_size=.33)
        X_valid, X_test, y_valid, y_test = train_test_split(X_hold, y_hold, train_size=.33)

        batch_size, train_loss_, valid_loss_ = 2048, [], []

        # Training Loop
        for _ in range(50):
            for b in range(batch_size, len(y_train), batch_size):
                X_batch, y_batch = X_train[b - batch_size:b], y_train[b - batch_size:b]
                classifier.partial_fit(X_batch, y_batch, classes=[0, 1])
                train_loss_.append(classifier.loss_)
                valid_loss_.append(log_loss(y_valid, classifier.predict_proba(X_valid)))

        plt.plot(range(len(train_loss_)), train_loss_, label="train loss")
        plt.plot(range(len(valid_loss_)), valid_loss_, label="validation loss")
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig('figures/justification_epochs_{}.png'.format(target), dpi=300, bbox_inches='tight')
        plt.show()

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

classifier = classifier = MLPClassifier(hidden_layer_sizes=(100, 50, 25, 13), #hidden_layers
                               activation='logistic', #sigmoid
                               solver='adam',
                               alpha=0.01, #regularization (maybe as second par?)
                               batch_size=2048,
                               learning_rate_init=0.001,
                               max_iter=500, #nr epochs
                               shuffle=True,
                               random_state=144,
                               tol = 0.0001, #optim tolerance (default used)
                               warm_start=True,
                               early_stopping=False, #set to true if want to validate (10%) within
                               beta_1=0.95,
                               beta_2=0.95,
                               n_iter_no_change=20 # default nr epochs for tol
                               )

epoch_test(data=data3, targets=['occf14b'], num_feat=num_feat, threshold=1, classifier=classifier)







