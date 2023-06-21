import pandas as pd
from ABC_algorithm_RF import ArtificialBeeColony
from localized_classmap import plotExplanations
from wrapper import AbstractWrapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, \
    confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import time
from joblib import parallel_backend
import joblib
import shap
shap.initjs()

def Trainer(X_train, y_train, pars):
    """
            :param X_train:     observed training data (df)
            :param y_train:     target training data (array)
            :param pars:        hyperparameter values (list)
            :return:            fitted classification model
    """

    pars = (np.array(pars) * scaling_factor)

    classifier = RandomForestClassifier(n_estimators=round(pars[0]),
                                        max_features=round(pars[1]),
                                        max_depth=round(pars[2]),
                                        min_samples_split=round(pars[3]),
                                        min_samples_leaf=round(pars[4]),
                                        n_jobs=-1,                   # run in parallel on 3 cores
                                        warm_start=True)  #reuse solution of previous call (speeds up about 2x)


    X_train = np.array(X_train)
    y_train = np.array(y_train)
    if X_train.ndim > 5: #adjust max dim size
        size = len(X_train)
        X_train = X_train.reshape(size, -1)

    with parallel_backend('threading', n_jobs=-1):
        clf = classifier.fit(X_train, y_train)

    return clf


def Tester(clf, X_test, y_test):
    """
            :param clf:     fitted classification model
            :param X_test:  observed test data (df)
            :param y_test:  target test data (array)
            :return:        the miscalculation value, true classifications, and predicted classifications
    """

    y_test = np.array(y_test)
    true = y_test

    X_test = np.array(X_test)

    if X_test.ndim > 5: # adjust max dim size
        size = len(X_test)
        X_test = X_test.reshape(size, -1)

    with parallel_backend('threading', n_jobs=-1):
        predicted = clf.predict(X_test)

    matthew = matthews_corrcoef(true, predicted)
    miscalc = 1 - matthew

    return miscalc, true, predicted

def SearchingPars(X_train, y_train, X_test, y_test, pars, feat_importance=False):
    """
            :param X_train:         observed training data (df)
            :param y_train:         target training data (array)
            :param X_test:          observed test data (df)
            :param y_test:          target test data (array)
            :param pars:            hyperparameter values (list)
            :param feat_importance: whether or not to calculate feature importance and class maps (binary)
            :return:                test results, feature importance, fitted model, and class map data
    """

    classification = Trainer(X_train, y_train, pars)

    if (feat_importance == True):
        with parallel_backend('threading', n_jobs=-1):
            explainer = shap.Explainer(classification)
            shap_test = explainer(X_test)
            importances = abs(shap_test[0].values)
            class0 = plotExplanations(classification, X_test, y_test.values, k=40, cl=0)
            class1 = plotExplanations(classification, X_test, y_test.values, k=40, cl=1)
            classmap = pd.concat([class0, class1], ignore_index=True)
    else:
        importances = None
        classmap = None

    return Tester(classification, X_test, y_test), importances, classification, classmap

def BestParMeasures(true, predicted):
    """
            :param true:        true classifications
            :param predicted:   predicted classifications
            :return:            performance metrics: accuracy, balanced accuracy, ROC score, F1 score,
                                confusion matrix, and matthews correlation coefficient
    """

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


class OptRFParameters(AbstractWrapper):

    def __init__(self, sub_X_train, sub_y_train, sub_X_test, sub_y_test):
        self.sub_X_train = sub_X_train
        self.sub_y_train = sub_y_train

        self.sub_X_test = sub_X_test
        self.sub_y_test = sub_y_test

        self.REPORT = ""


    def objective_function_value(self, decision_variable_values):
        """
                :param decision_variable_values:     hyperparameter values (list)
                :return:                             test results, feature importance, fitted model, and class map data
        """

        return SearchingPars(self.sub_X_train, self.sub_y_train,
                             self.sub_X_test, self.sub_y_test,
                             decision_variable_values)


def Search(sub_X_train, sub_y_train, sub_X_test, sub_y_test):
    """
            :param sub_X_train:     subset of observed training data
            :param sub_y_train:     subset of target training data
            :param sub_X_test:      subset of observed test data
            :param sub_y_test:      subset of target test data
            :return:                optimal parameters and corresponding matthews coefficient
    """


    # adjust objective function
    func = OptRFParameters(sub_X_train, sub_y_train, sub_X_test, sub_y_test)



    # order: n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf
    min_hyperpars = [1, 2, 2, 2, 1]
    max_hyperpars = [500,np.array(sub_X_train).shape[1]-1, 50, 10, 10]
    list_pars = [min_hyperpars, max_hyperpars]


    # scale the hyperparameters
    global scaling_factor
    scaling_factor = np.linalg.norm(list_pars)
    list_par_norm = (list_pars / scaling_factor)

    abc = ArtificialBeeColony(function=func,
                              l_bound=list_par_norm[0],
                              u_bound=list_par_norm[1],
                              hive_size=50,
                              max_iterations=50,
                              seed=144,
                              verbose=False,
                              )

    pars_scaled = abc.solution
    pars = np.array(pars_scaled) * scaling_factor

    fitness = abc.best_place

    message = "Best parameters = {} \nBest evaluation value (Matthews misclassification rate) = {} "
    print(message.format(list(pars.round().tolist()), float(fitness)))

    return pars, fitness

def KfoldCV(k, X, y, filename, features):
    """
            Perform k-fold cross-validation on the parameter searching/optimization process.

            :param k:           the number of folds (int)
            :param X:           the complete (subset of) observed data set (df)
            :param y:           the target set (array)
            :param filename:    the name of the file (str)
            :param features:    the list of features present in the observed data
    """

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
        print("\n\nFold number = "+str(i+1))

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
        par_scaled = pars['model {}'.format(m)] / scaling_factor

        for f in range(0, k):
            X_train_sub = (X_train.iloc[train_ind[f]])
            y_train_sub = (y_train.iloc[np.array(train_ind[f])])

            X_validate = (X_train.iloc[np.array(test_ind[f])])
            y_validate = (y_train.iloc[np.array(test_ind[f])])

            res_fold = SearchingPars(X_train_sub, y_train_sub, X_validate, y_validate, par_scaled)

            res_fold_measures = BestParMeasures(res_fold[0][1],  # true
                                                res_fold[0][2])  # predicted

            measures['fold {}'.format(f)] = list(res_fold_measures)

        # print(measures)

        df_measures = pd.DataFrame.from_dict(measures, orient='index')
        df_measures.drop(columns=4, inplace=True)
        averages['model {}'.format(m)] = df_measures.mean().tolist()

    df_averages = pd.DataFrame.from_dict(averages, orient='index')
    df_averages.columns = ['accuracy', 'balanced accuracy', 'ROC', 'F1','Matthews']
    best = df_averages['Matthews'].idxmax()

    df_pars = pd.DataFrame.from_dict(pars, orient='index').round()
    df_pars.columns = ['number estimators', 'max features', 'max depth', 'min split', 'min leaf']

    df_results = pd.concat([df_pars, df_averages], ignore_index=False, axis=1, sort=False)

    # store important info and continue test phase
    best_pars = df_pars.loc[[best]].values.flatten().tolist()
    best_pars_scaled = best_pars / scaling_factor

    ## test phase
    res2 = SearchingPars(X_train, y_train, X_test, y_test, best_pars_scaled, feat_importance=True) #adjust pars

    res2_measures = BestParMeasures(res2[0][1], res2[0][2])
    res2_feat_importance = res2[1]

    importances = pd.DataFrame(res2_feat_importance, index=features)
    classmaps = res2[3]
    importances.to_csv("feature importance/RF/ft_importancesSHAP_{}.csv".format(filename))
    classmaps.to_csv("classmaps/RF/cm_{}.csv".format(filename))

    # save best model
    best_model = res2[2]
    joblib.dump(best_model, "models/RF/best_{}.joblib".format(filename), compress=3)

    # add test results to df results
    res = list(res2_measures)
    del res[4]
    df_results.loc['best model'] = [*best_pars, *res]

    # save the results
    df_results.to_csv("parameter_search_outputs/RF/{}.csv".format(filename))

    # use following command to load the model (these are notes for later)
    # loaded_rf = joblib.load("best_random_forest.joblib")

    elapsed_t = time.process_time() - t
    print(f"Elapsed time test phase: {elapsed_t} sec")

