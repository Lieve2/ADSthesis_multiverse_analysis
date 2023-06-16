import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from ABC_algorithm_MLP import ArtificialBeeColony
from localized_classmap import plotExplanations
from wrapper import AbstractWrapper
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, \
    confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import time
from joblib import parallel_backend
import shap
shap.initjs()

import warnings
warnings.filterwarnings("ignore")

def Trainer(X_train, y_train, pars):

    # nr_units int, dropout_rate 0-1float, learning_rate 0-01float,
    # beta_1 0-1float, beta_2 0-1float
    nr_units = pars[0] * scaling_factor
    hidden_layers = (round(nr_units), round(nr_units/2), round(nr_units/4), round(nr_units/8)) # 4 hidden layers

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    if X_train.ndim > 5:
        size = len(X_train)
        X_train = X_train.reshape(size, -1)

    classifier = MLPClassifier(hidden_layer_sizes=hidden_layers, #hidden_layers
                               activation='logistic', #sigmoid
                               solver='adam',
                               alpha=pars[1], #regularization
                               batch_size=2048,
                               learning_rate_init=pars[2],
                               max_iter=150, #nr epochs
                               shuffle=True,
                               random_state=144,
                               tol = 0.0001, #optim tolerance (default used)
                               warm_start=True,
                               early_stopping=True, #set to true if want to validate (10%) within
                               beta_1=pars[3],
                               beta_2=pars[4],
                               n_iter_no_change=20 # default nr epochs for tol
                               )

    with parallel_backend('threading', n_jobs=-1):
        classifier.fit(X_train, y_train)

    return classifier

def Tester(clf, X_test, y_test):

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

    classification = Trainer(X_train, y_train, pars)

    if (feat_importance == True):
        with parallel_backend('threading', n_jobs=-1):
            X_train_sum = shap.sample(X_train, 300, random_state=144)
            X_test_sum = shap.sample(X_test, 500, random_state=144)

            explainer = shap.KernelExplainer(classification.predict_proba, X_train_sum)
            shap_test = explainer.shap_values(X_test_sum)
            importances = np.mean(np.abs(shap_test),
                                  axis=1).T
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

class OptRFParameters(AbstractWrapper):

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

    # adjust objective function
    func = OptRFParameters(sub_X_train, sub_y_train, sub_X_test, sub_y_test)


    # order: nr_units, L2 regu term (alpha), learning_rate, beta_1, beta_2
    min_hyperpars = [20, 0.00001, 0.0001, 0.6, 0.6]
    max_hyperpars = [200, 0.01, 0.01, 0.999, 0.999]
    list_pars = [min_hyperpars, max_hyperpars]

    # scale the hyperparameters
    global scaling_factor
    to_be_scaled = [list_pars[0][0], list_pars[1][0]]
    scaling_factor = np.linalg.norm(to_be_scaled)
    list_par_norm = ([list_pars[0][0], list_pars[1][0]] / scaling_factor) #.tolist()

    final_list_pars = [[list_par_norm[0], *list_pars[0][1:]], [list_par_norm[1], *list_pars[1][1:]]]

    abc = ArtificialBeeColony(function=func,
                              l_bound=final_list_pars[0] ,
                              u_bound=final_list_pars[1] ,
                              hive_size=50,
                              max_iterations=50,
                              seed=144,
                              verbose=False,
                              )

    pars_scaled = abc.solution
    pars_unsc = pars_scaled[0] * scaling_factor
    pars = [pars_unsc, *pars_scaled[1:]]

    fitness = abc.best_place

    message = "Best parameters = {} \nBest evaluation value (balanced misclassification rate) = {} "
    print(message.format(list([round(pars[0]), *pars[1:]]), float(fitness)))

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
        print("\n\nFold number = " + str(i + 1))

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
        par_sc = pars['model {}'.format(m)][0] /scaling_factor
        par_scaled = [par_sc, *pars['model {}'.format(m)][1:]]

        for f in range(0, k):
            X_train_sub = (X_train.iloc[train_ind[f]])
            y_train_sub = (y_train.iloc[np.array(train_ind[f])])

            X_validate = (X_train.iloc[np.array(test_ind[f])])
            y_validate = (y_train.iloc[np.array(test_ind[f])])

            res_fold = SearchingPars(X_train_sub, y_train_sub, X_validate, y_validate, par_scaled)

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
    df_pars.columns = ['number units', 'L2 regularization', 'learning rate', 'beta 1', 'beta 2']

    df_results = pd.concat([df_pars, df_averages], ignore_index=False, axis=1, sort=False)

    # store important info and continue test phase
    best_pars = df_pars.loc[[best]].values.flatten().tolist()
    par_sc2 = best_pars[0] / scaling_factor
    best_pars_sc = [par_sc2, *best_pars[1:]]

    ## test phase (train on whole training set test with left out test set)
    res2 = SearchingPars(X_train, y_train, X_test, y_test, best_pars_sc, feat_importance=True)

    res2_measures = BestParMeasures(res2[0][1], res2[0][2])
    res2_feat_importance = res2[1]

    importances = pd.DataFrame(res2_feat_importance, index=features)
    classmaps = res2[3]
    importances.to_csv("feature importance/MLP/ft_importancesSHAP_{}.csv".format(filename))
    classmaps.to_csv("classmaps/MLP/cm_{}.csv".format(filename))

    # save best model
    best_model = res2[2]
    joblib.dump(best_model, "models/MLP/best_{}.joblib".format(filename), compress=3)

    # add test results to df results
    res = list(res2_measures)
    del res[4]
    df_results.loc['best model'] = [*best_pars_sc, *res]

    # save the results
    df_results.to_csv("parameter_search_outputs/MLP/{}.csv".format(filename))

    # use following command to load the model
    # loaded_rf = joblib.load("best_random_forest.joblib")

    elapsed_t = time.process_time() - t
    print(f"Elapsed time test phase: {elapsed_t} sec")