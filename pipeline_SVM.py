import joblib
from sklearn.ensemble import BaggingClassifier
from ABC_algorithm_SVM import ArtificialBeeColony
from localized_classmap import plotExplanations
from wrapper import AbstractWrapper
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, \
    confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
from joblib import parallel_backend
import numpy as np
import pandas as pd
import time
import shap
shap.initjs()

import warnings
warnings.filterwarnings("ignore")



def Trainer(X_train, y_train, pars):

    # classifier = SVC(C=pars[0], gamma=pars[1])
    classifier = BaggingClassifier(base_estimator=SVC(C=pars[0], gamma=pars[1], probability=True), #changed prob
                                   n_estimators=20, random_state=144,
                                   n_jobs=-1, max_samples=0.05, bootstrap=False)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    if X_train.ndim > 2:
        size = len(X_train)
        X_train = X_train.reshape(size, -1)

    with parallel_backend('threading', n_jobs=-1):
        clf = classifier.fit(X_train, y_train)

    return clf

def Tester(clf, X_test, y_test):

    y_test = np.array(y_test)
    X_test = np.array(X_test)

    true = y_test

    if X_test.ndim > 2:
        size = len(X_test)
        X_test = X_test.reshape(size, -1)

    with parallel_backend('threading', n_jobs=-1):
        predicted = clf.predict(X_test)

    matthew = matthews_corrcoef(true, predicted)
    miscalc = 1 - matthew

    return miscalc, true, predicted



def SearchingPars(X_train, y_train, X_test, y_test, pars, feat_importance=False):

    classification = Trainer(X_train, y_train, pars)

    if feat_importance==True:
        with parallel_backend('threading', n_jobs=-1):

            # set to 75, 35
            X_train_sum = shap.sample(X_train, 75, random_state=144)
            X_test_sum = shap.sample(X_test, 35, random_state=144)

            explainer = shap.KernelExplainer(classification.predict_proba, X_train_sum)
            shap_test = explainer.shap_values(X_test_sum)
            importances = np.mean(np.abs(shap_test), axis=1).T

            class0 = plotExplanations(classification, X_test, y_test.values, k=40, cl=0)
            class1 = plotExplanations(classification, X_test, y_test.values, k=40, cl=1)
            classmap = pd.concat([class0, class1], ignore_index=True)

    else:
        importances = None
        classmap = None

    return Tester(classification, X_test, y_test), importances, classification, classmap

def BestParMeasures(true, predicted):

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
    matthews = matthews_corrcoef(true, predicted)

    return acc, bal_acc, ROC, f1, conf_matrix, matthews

class OptSVMParameters(AbstractWrapper):

    def __init__(self, sub_X_train, sub_y_train, sub_X_test, sub_y_test):
        self.sub_X_train = sub_X_train
        self.sub_y_train = sub_y_train

        self.sub_X_test = sub_X_test
        self.sub_y_test = sub_y_test

        self.REPORT = ""

    def objective_function_value(self, decision_variable_values):
        return SearchingPars(self.sub_X_train, self.sub_y_train,
                             self.sub_X_test, self.sub_y_test,
                             decision_variable_values)


def Search(sub_X_train, sub_y_train, sub_X_test, sub_y_test):
    """Find..."""

    func = OptSVMParameters(sub_X_train, sub_y_train, sub_X_test, sub_y_test)

    # set ranges of hyperparameters, order: C, gamma
    min_hyperpars = [0.0001, 0]
    max_hyperpars = [1, 1]

    # if wanting to add different kernels later: use scaling idea to encode and decode them

    abc = ArtificialBeeColony(function=func,
                              l_bound=min_hyperpars,
                              u_bound=max_hyperpars,
                              hive_size=50,
                              max_iterations=50,
                              seed=144,
                              verbose=False,
                              )

    pars = abc.solution
    fitness = abc.best_place


    message = "Best parameters = {} \nBest evaluation value (Matthews misclassification rate) = {} "
    print(message.format(list(pars), float(fitness)))

    return pars, fitness

def KfoldCV(k, X, y, filename, features):
    """Perform k-fold cross-validation on the parameter searching/optimization process.
    Takes the following parameters as input:

    k: the number of folds
    X: the complete data set
    y: the target set """

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=144)

    # train-validation split
    stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=144)
    folds = stratified_kfold.split(X_train, y_train)

    # order: n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf
    train_ind = []
    test_ind = []
    pars = {}
    measures = {}
    averages = {}

    # store train and test data for each fold
    for i, (train_index, test_index) in enumerate(folds):
        print("\n\nFold number = " + str(i+1))

        train_ind.append(train_index.tolist())
        test_ind.append(test_index.tolist())

        # timer search progress - start
        t = time.process_time()

        X_train_sub = (X_train.iloc[train_ind[i]])
        y_train_sub = (y_train.iloc[np.array(train_ind[i])])

        X_validate = (X_train.iloc[np.array(test_ind[i])])
        y_validate = (y_train.iloc[np.array(test_ind[i])])

        # perform parameter search on train and test split of current fold #measures=measures
        par, fit = Search(X_train_sub, y_train_sub, X_validate, y_validate)

        pars['model {}'.format(i)] = par

        elapsed_t = time.process_time() - t
        print(f"Elapsed time optimal parameter search: {elapsed_t} sec")

    # 3x3 (3 folds, 3 models)
    for m in range(0, k):
        par_m = pars['model {}'.format(m)]

        for f in range(0, k):
            X_train_sub = (X_train.iloc[train_ind[f]])
            y_train_sub = (y_train.iloc[np.array(train_ind[f])])

            X_validate = (X_train.iloc[np.array(test_ind[f])])
            y_validate = (y_train.iloc[np.array(test_ind[f])])

            res_fold = SearchingPars(X_train_sub, y_train_sub, X_validate, y_validate, par_m)

            res_fold_measures = BestParMeasures(res_fold[0][1],  # true
                                                res_fold[0][2])  # predicted

            measures['fold {}'.format(f)] = list(res_fold_measures)

        df_measures = pd.DataFrame.from_dict(measures, orient='index')
        df_measures.drop(columns=4, inplace=True)
        averages['model {}'.format(m)] = df_measures.mean().tolist()

    df_averages = pd.DataFrame.from_dict(averages, orient='index')
    df_averages.columns = ['accuracy', 'balanced accuracy', 'ROC', 'F1', 'Matthews']
    best = df_averages['Matthews'].idxmax()

    df_pars = pd.DataFrame.from_dict(pars, orient='index')
    df_pars.columns = ['C', 'gamma']

    df_results = pd.concat([df_pars, df_averages], ignore_index=False, axis=1, sort=False)

    # store important info and continue test phase
    best_pars = df_pars.loc[[best]].values.flatten().tolist()


    ## test phase (train on whole training set test with left out test set)
    res2 = SearchingPars(X_train, y_train, X_test, y_test, best_pars, feat_importance=True)  # adjust pars

    res2_measures = BestParMeasures(res2[0][1], res2[0][2])
    res2_feat_importance = res2[1]

    importances = pd.DataFrame(res2_feat_importance, index=features)
    classmaps = res2[3]
    importances.to_csv("feature importance/SVM/ft_importancesSHAP_{}.csv".format(filename))
    classmaps.to_csv("classmaps/SVM/cm_{}.csv".format(filename))

    # save best model
    best_model = res2[2]
    joblib.dump(best_model, "models/SVM/best_{}.joblib".format(filename), compress=3)

    # add test results to df results
    res = list(res2_measures)
    del res[4]
    df_results.loc['best model'] = [*best_pars, *res]

    # save the results
    df_results.to_csv("parameter_search_outputs/SVM/{}.csv".format(filename))

    # use following command to load the model (these are notes for later)
    # loaded_rf = joblib.load("best_random_forest.joblib")

    elapsed_t = time.process_time() - t
    print(f"Elapsed time test phase: {elapsed_t} sec")





