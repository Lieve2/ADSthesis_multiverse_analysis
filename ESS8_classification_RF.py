
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

from pipeline_RF import KfoldCV, KfoldCVPCA #!!!!!!

import scipy.spatial.distance as ssd


def encode_scale_data(data, tuning_target):
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
    X = data.drop(tuning_target, axis=1).fillna(-999)

    # scale the data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    return X, y

def pca(X, y, figure_name, training=True, verbose=False):

    pca = PCA(n_components=X.shape[1])

    data_orig_new_pca = pca.fit_transform(X)

    perc_explained = pca.explained_variance_ / (np.sum(pca.explained_variance_))
    cumu_explained = np.cumsum(perc_explained)

    if (verbose == True):
        plt.plot(cumu_explained)
        plt.grid()
        plt.xlabel('number of components')
        plt.ylabel('% variance explained')

        plt.savefig('figures/expl_var_{}_original.png'.format(figure_name), dpi=200, bbox_inches='tight')

    # split into train and test data for PCA
    if (training == True):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=144)

        pca_post = PCA(n_components=232)  # index+1
        pca_train = pca_post.fit_transform(X_train)
        pca_test = pca_post.transform(X_test)

        df_train_pca = pd.DataFrame(pca_train)

        corr_post = df_train_pca.corr()

    else:
        #maybe run this with parallel processing?
        pca_post = PCA(n_components=232)  # index+1
        pca_total = pca_post.fit_transform(X)

        df_total_pca = pd.DataFrame(pca_total)

        corr_post = df_total_pca.corr()

    if (verbose == True):
        plt.figure(figsize=(48, 36))

        # undefined correlation (does not vary, i.e. no missingness at all)
        color = plt.get_cmap('RdYlGn')  # default color
        color.set_bad('lightblue')  # if the value is bad the color would be lightblue instead of white

        # generate heatmap
        sns.heatmap(corr_post, annot=False, vmin=-1, vmax=1, cmap=color)  # mask=mask,
        plt.title('Correlation Coefficient Of Predictors after PCA')
        plt.savefig('figures/corr_coef_postPCA_{}.png'.format(figure_name), dpi=200, bbox_inches='tight')
        #plt.show()

    if (training==True):
        return X_train, X_test, y_train, y_test, pca_post
    else:
        return X, y, pca_post

def extract_feature_importance(data, tuning_target, pca_post, filename):

    # extract feature names original data
    feature_names = data.drop(tuning_target, axis=1).columns

    # # load permutation importance file
    # df_perm_imp = pd.read_csv(file)
    # perm_imps = np.array(df_perm_imp.iloc[:,1]).reshape(-1,1)

    # retrieve eigenvectors
    pca_eigenvec = pca_post.components_

    # calculate 'relative'/absolute importance over all PC's
    rel_imp = np.abs(pca_eigenvec) #* perm_imps

    # calculate total importance per feature
    total_imp = np.sum(rel_imp, axis=0).reshape(1,-1)

    df_feature_importance = pd.DataFrame(total_imp, columns=feature_names)

    df_feature_importance.to_csv('results/total_feat_imp_{}.csv'.format(filename))

    return df_feature_importance


def hyperparameter_tuningPCA(data, tuning_target, filename, PCA_transformed=True):

    X, y = encode_scale_data(data, tuning_target)

    X_train, X_test, y_train, y_test, pca_post = pca(X=X, y=y, figure_name='tuningRF', verbose=True)

    #print(X_train.shape, y_train.shape)

    # timer - start
    t = time.process_time()

    # perform k-fold CV
    KfoldCVPCA(3, X_train, X_test, y_train, y_test, filename=filename, PCA_transformed=PCA_transformed)

    # timer - end
    elapsed_time = time.process_time() - t
    print(f"Elapsed time for K-fold CV: {elapsed_time} sec")

    extract_feature_importance(data=data,
                               tuning_target=tuning_target,
                               pca_post=pca_post,
                               filename=filename)


def tester(classifier, X_test, y_test):

    y_test = np.array(y_test)
    true = y_test

    X_test = np.array(X_test)

    if X_test.ndim > 5:
        size = len(X_test)
        X_test = X_test.reshape(size, -1)

    with parallel_backend('threading', n_jobs=3):
        predicted = classifier.predict(X_test)

        return true, predicted


def extract_performance_measures(true, predicted):

    true = np.array(true)
    predicted = np.array(predicted)

    acc = accuracy_score(true, predicted)
    bal_acc = balanced_accuracy_score(true, predicted)
    try:
        ROC = roc_auc_score(true, predicted)
    except ValueError:
        ROC = np.nan
    f1 = f1_score(true, predicted)
    conf_matrix = confusion_matrix(true, predicted)

    return acc, bal_acc, ROC, f1, conf_matrix



def run_model(data, model, targets, filename):

    """Run the hyperparameter tuned model on the list of targets
    and for each target/run extract the feature importances and record the important info
    as done in the KfoldCV function"""

    # initialize variables for storing info
    REPORT = ""
    REPORT_CSV = ""

    t = time.process_time()

    for target in targets:

        t = time.process_time()

        # scale and split data
        X, y = encode_scale_data(data, target)
        X_pca, y_pca, pca_post = pca(X=X, y=y,
                                     figure_name='RF_{}'.format(target),
                                     training=False, verbose=False)

        elapsed_t = time.process_time() - t
        print(f"Elapsed time PCA {target}: {elapsed_t} sec")

        # initialize timer
        t = time.process_time()

        true, predicted = tester(model, X_pca, y_pca)
        metrics = extract_performance_measures(true, predicted)
        extract_feature_importance(data, target, pca_post, 'RF_{}'.format(target))

        elapsed_t = time.process_time() - t
        print(f"Elapsed time test run {target}: {elapsed_t} sec")

        # record metrics info
        REPORT += "Target: " + str(target) + "\n"
        REPORT += "Accuracy: " + str(metrics[0]) + "\n"
        REPORT += "Balanced accuracy: " + str(metrics[1]) + "\n"
        REPORT += "ROC score: " + str(metrics[2]) + "\n"
        REPORT += "F1 score: " + str(metrics[3]) + "\n"
        REPORT += "confusion matrix: " + "\n" + str(metrics[4]) + "\n"
        REPORT += "Elapsed time running tester: " + "\n" + str(round(elapsed_t, 4)) + "\n\n\n"

        t = time.process_time()

        REPORT_CSV += str(target) + ","
        REPORT_CSV += str(metrics[0]) + ","
        REPORT_CSV += str(metrics[1]) + ","
        REPORT_CSV += str(metrics[2]) + ","
        REPORT_CSV += str(metrics[3]) + ","
        REPORT_CSV += str(metrics[4]) + "\n"

        elapsed_t = time.process_time() - t
        print(f"Elapsed time recording data {target}: {elapsed_t} sec")

    text_file = open("results/{}.txt".format(filename), "w")
    text_file.write(REPORT)

    columns = "Target, Accuracy, Balanced accuracy, ROC, F1, Confusion matrix"
    csv_file = open("results/{}.csv".format(filename), "w")
    csv_file.write(columns + '\n' + REPORT_CSV)

    elapsed_t = time.process_time() - t
    print(f"Total elapsed time: {elapsed_t} sec")


## ----- hyperpar without PCA ----- ##

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
    # corr = np.array(X.corr('spearman'))
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
    #data_obj = data.select_dtypes(include=object)
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

def hyperparameter_tuning(data, targets, threshold, filename, num_feat, PCA_transformed=False):

    for target in targets:

        X, y, features, clusters = encode_scale_data_perm(data, target, threshold, num_feat)
        # print(features)

        df_clusters = pd.DataFrame.from_dict(clusters, orient='index')

        df_clusters.to_csv('clusters info/RF/{}_{}.csv'.format(filename, target))

        # timer - start
        t = time.process_time()

        # perform k-fold CV
        KfoldCV(3, X, y,
                filename='{}_{}'.format(filename, target),
                PCA_transformed=PCA_transformed,
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

## ----- run hyperparameter tuning ----- ## --> maybe add this to another file for better overview :)
#hyperparameter_tuningPCA(data, tuning_target='inwehh', filename='RF_tuning', PCA_transformed=True)

## ----- test functions except tuning itself ----- ##

# X, y = encode_scale_data(data, 'inwehh')
# pca_post = pca(data, X, y, 'test')[4]
# extract_feature_importance(data, tuning_target='inwehh', pca_post=pca_post, filename='test')

## ----- run model on all desired targets ----- ##

# load model
# model_rf = joblib.load("models/best_random_forest.joblib")
# targets = data.columns
# run_model(data, model_rf, targets, 'targets_RF')

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

hyperparameter_tuning(data=data3, targets=targets3, num_feat=num_feat, threshold=1, filename='RF')









