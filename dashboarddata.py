import glob
import os.path
import numpy as np
import pandas
import pandas as pd
from utility_functions import multi_csv_to_df

pandas.set_option('display.max_rows', 188)


# ----- extract feature info data ----- #

df_featureinfo = pd.read_csv('dashboardapp/finaldata/feature_info.csv', sep=';')

# ----- make importance test data set ----- #

ran = range(1, 20)
c = ['Feature {0}'.format(s) for s in ran]

vals = np.random.rand(19)
vals2 = np.random.rand(19)
vals3 = np.random.rand(19)

test_data = pd.DataFrame(vals, columns=['target X'], index=c)
test_data2 = pd.DataFrame(vals2, columns=['target Y'], index=c)
test_data3 = pd.DataFrame(vals3, columns=['target Z'], index=c)

df_importancem1_test = pd.concat([test_data, test_data2, test_data3], axis='columns')
df_importancem1_test['average'] = df_importancem1_test.mean(axis=1)  # calculate average per row

# ----- make performance data set ----- #

c = ['target X', 'target Y', 'target Z']

vals = (np.random.rand(3)).tolist()
vals2 = (np.random.rand(3)).tolist()
vals3 = (np.random.rand(3)).tolist()
vals4 = (np.random.rand(3)).tolist()
vals5 = (np.random.rand(3)).tolist()
vals6 = (np.random.rand(3)).tolist()

test_data1 = pd.DataFrame(np.array([vals, vals2]).T, columns=['accuracy', 'balanced accuracy'], index=c)
test_data1 = test_data1.T
test_data1['average'] = test_data1.mean(axis=1)

# ----- make importance data set for second and third model ----- #

ran = range(1, 20)
c = ['Feature {0}'.format(s) for s in ran]

test_data = pd.DataFrame(vals, columns=['target X'], index=c)
test_data2 = pd.DataFrame(vals2, columns=['target Y'], index=c)
test_data3 = pd.DataFrame(vals3, columns=['target Z'], index=c)

df_importancem2_test = pd.concat([test_data, test_data2, test_data3], axis='columns')
df_importancem2_test['average'] = df_importancem2_test.mean(axis=1)  # calculate average per row

ran = range(1, 20)
c = ['Feature {0}'.format(s) for s in ran]

vals = np.random.rand(19)
vals2 = np.random.rand(19)
vals3 = np.random.rand(19)

test_data = pd.DataFrame(vals, columns=['target X'], index=c)
test_data2 = pd.DataFrame(vals2, columns=['target Y'], index=c)
test_data3 = pd.DataFrame(vals3, columns=['target Z'], index=c)

df_importancem3_test = pd.concat([test_data, test_data2, test_data3], axis='columns')
df_importancem3_test['average'] = df_importancem3_test.mean(axis=1)  # calculate average per row


# ----- make performance data set for 2nd and 3rd model ----- #
c = ['target X', 'target Y', 'target Z']

vals = (np.random.rand(3)).tolist()
vals2 = (np.random.rand(3)).tolist()
vals3 = (np.random.rand(3)).tolist()

test_data2 = pd.DataFrame(np.array([vals3, vals4]).T, columns=['accuracy', 'balanced accuracy'], index=c)
test_data3 = pd.DataFrame(np.array([vals5, vals6]).T, columns=['accuracy', 'balanced accuracy'], index=c)

test_data2 = test_data2.T
test_data2['average'] = test_data2.mean(axis=1)

test_data3 = test_data3.T
test_data3['average'] = test_data3.mean(axis=1)

df_importancem1_test.insert(0, 'Feature', df_importancem1_test.index)
df_importancem2_test.insert(0, 'Feature', df_importancem2_test.index)
df_importancem3_test.insert(0, 'Feature', df_importancem3_test.index)

test_data1.insert(0, 'Metric', test_data1.index)
test_data2.insert(0, 'Metric', test_data2.index)
test_data3.insert(0, 'Metric', test_data3.index)

df_importancem1_test.reset_index(inplace=True, drop=True)
df_importancem2_test.reset_index(inplace=True, drop=True)
df_importancem3_test.reset_index(inplace=True, drop=True)

test_data1.reset_index(inplace=True, drop=True)
test_data2.reset_index(inplace=True, drop=True)
test_data3.reset_index(inplace=True, drop=True)

# store test data in csv files
df_importancem1_test.to_csv("dashboardapp/testdata/importance_model1.csv", index=False)
df_importancem2_test.to_csv("dashboardapp/testdata/importance_model2.csv", index=False)
df_importancem3_test.to_csv("dashboardapp/testdata/importance_model3.csv", index=False)

test_data1.to_csv("dashboardapp/testdata/performance_model1.csv", index=False)
test_data2.to_csv("dashboardapp/testdata/performance_model2.csv", index=False)
test_data3.to_csv("dashboardapp/testdata/performance_model3.csv", index=False)

# ----- make class maps test data ----- #
df_cm_test_target0 = pd.read_csv('classmaps/MLP/cm_MLP_netustm.csv')

df_cm_test1 = pd.read_csv('classmaps/MLP/cm_MLP_netustm.csv')
df_cm_test2 = pd.read_csv('classmaps/MLP/cm_MLP_netustm.csv')
df_cm_test3 = pd.read_csv('classmaps/MLP/cm_MLP_netustm.csv')

df_cm_test1.to_csv('dashboardapp/testdata/cm_target_X.csv', index=False)
df_cm_test2.to_csv('dashboardapp/testdata/cm_target_Y.csv', index=False)
df_cm_test3.to_csv('dashboardapp/testdata/cm_target_Z.csv', index=False)


### --------- Final dashboard data --------- ###

# load data
data = pd.read_csv('ESS8 data/ESS8_subset_cleaned_timeadj_wmissingvals.csv', low_memory=False)

# make df with missingness percentage of features with missingness
missing_cols = data.columns[data.isnull().any()].to_list()
percent_missing = data[missing_cols].isnull().sum() * 100 / len(data)
missing_info = pd.DataFrame({'column name': missing_cols,
                             'percentage missing': percent_missing})

# extract rows with > 5% missing (74 features)
many_missing = missing_info[missing_info['percentage missing'] > 5]
targets = many_missing['column name'].tolist()
targets_sorted = sorted(targets)

### ----- feature importance RF ----- ###

path_imp_RF = str(os.getcwd())+'/feature importance/RF/'
feat_imp_RF = glob.glob(os.path.join(path_imp_RF, '*.csv'))

df_feat_imp_RF = multi_csv_to_df(feat_imp_RF, axis=1, index_col='Unnamed: 0')
df_feat_imp_RF = df_feat_imp_RF.iloc[:, ::2]
df_feat_imp_RF.columns = targets_sorted

df_feat_imp_RF.insert(loc=0, column='Feature', value=df_feat_imp_RF.index)
df_feat_imp_RF.reset_index(drop=True, inplace=True)

df_feat_imp_RF.to_csv("dashboardapp/finaldata/importance_RF.csv", index=True)  # features


### ----- model performance RF ----- ###

path_perf_RF = str(os.getcwd())+'/parameter_search_outputs/RF/'
perf_RF = glob.glob(os.path.join(path_perf_RF, '*.csv'))

df_perf_RF = multi_csv_to_df(perf_RF, axis=0, index_col=None)
df_perf_RF.rename(columns={'Unnamed: 0': 'model'}, inplace=True)
df_perf_RF = df_perf_RF.loc[df_perf_RF['model'] == 'best model'].reset_index(drop=True)
df_perf_RF.index = targets_sorted
df_perf_RF.loc['average'] = df_perf_RF.mean(axis=0, numeric_only=True)

df_perf_RF = df_perf_RF.T
df_perf_RF = df_perf_RF.drop(['model', 'number estimators',
                              'max features', 'max depth',
                              'min split', 'min leaf'], axis=0)
df_perf_RF.insert(loc=0, column='Metric', value=df_perf_RF.index)
df_perf_RF.reset_index(drop=True, inplace=True)

df_perf_RF.to_csv("dashboardapp/finaldata/performance_RF.csv", index=False)

### ----- cluster data RF + average feature importance ----- ###

pandas.set_option('display.max_rows', 188)

# make copy of original feature importance df of all targets
df_feat_imp_RF_c = df_feat_imp_RF.copy()

for target in targets_sorted:
    df_cluster_info = pd.read_csv('clusters info/RF/RF_{}.csv'.format(target))
    df_cluster_info.rename(columns={'Unnamed: 0': 'cluster number'}, inplace=True)
    df_cluster_info.set_index(df_cluster_info.columns[0], inplace=True)

    # drop target from list of feature names
    cols = data.columns.drop(target)

    # replace numbers with feature names and add empty column for iteration process later
    df_cluster_info.replace(to_replace=list(range(0, len(data.columns)-1)), value=cols, inplace=True)
    df_cluster_info['empty'] = np.nan

    # set row counter
    r = 0

    # iterate through list of features present in selected subset
    for i in df_cluster_info.iloc[:, 0]:

        # find corresponding feature importance
        imp = df_feat_imp_RF_c.loc[df_feat_imp_RF_c['Feature'] == i, target]

        # set column counter
        c = 1

        # for the features in each cluster
        while not pd.isna(df_cluster_info.iloc[r, c]):

            # extract the c-th feature in a particular cluster (r)
            feature = df_cluster_info.iloc[r, c]

            # check if feature is present in feature importance data
            try:
                row = df_feat_imp_RF_c[df_feat_imp_RF_c['Feature'] == feature].index[0]
                # print(row)
                df_feat_imp_RF_c.at[row, target] = imp  # represents first target

            # if not, add it to the end of the df
            except:
                df_feat_imp_RF_c.loc[len(df_feat_imp_RF_c)] = {'Feature': feature, target: float(imp)}

            # move onto the next column (feature)
            c += 1
        # move onto the next row (feature present in selected subset)
        r += 1

# calculate average importance per feature and add it to the df
df_feat_imp_RF_c['average'] = df_feat_imp_RF_c.mean(axis=1, skipna=True, numeric_only=True)

# store data as csv
df_feat_imp_RF_c.to_csv("dashboardapp/finaldata/complete_importance_RF.csv", index=False)

### ----- feature importance SVM ----- ###

path_imp_SVM = str(os.getcwd())+'/feature importance/SVM/'
feat_imp_SVM = glob.glob(os.path.join(path_imp_SVM, '*.csv'))

df_feat_imp_SVM = multi_csv_to_df(feat_imp_SVM, axis=1, index_col='Unnamed: 0')
df_feat_imp_SVM = df_feat_imp_SVM.iloc[:, ::2]
df_feat_imp_SVM.columns = targets_sorted

df_feat_imp_SVM.insert(loc=0, column='Feature', value=df_feat_imp_SVM.index)
df_feat_imp_SVM.reset_index(drop=True, inplace=True)

df_feat_imp_SVM.to_csv("dashboardapp/finaldata/importance_SVM.csv", index=True)

### ----- model performance SVM ----- ###

path_perf_SVM = str(os.getcwd())+'/parameter_search_outputs/SVM/'
perf_SVM = glob.glob(os.path.join(path_perf_SVM, '*.csv'))

df_perf_SVM = multi_csv_to_df(perf_SVM, axis=0, index_col=None)
df_perf_SVM.rename(columns={'Unnamed: 0': 'model'}, inplace=True)
df_perf_SVM = df_perf_SVM.loc[df_perf_SVM['model'] == 'best model'].reset_index(drop=True)
df_perf_SVM.index = targets_sorted
df_perf_SVM.loc['average'] = df_perf_SVM.mean(axis=0, numeric_only=True)

df_perf_SVM = df_perf_SVM.T
df_perf_SVM = df_perf_SVM.drop(['model', 'C', 'gamma'], axis=0)

df_perf_SVM.insert(loc=0, column='Metric', value=df_perf_SVM.index)
df_perf_SVM.reset_index(drop=True, inplace=True)

df_perf_SVM.to_csv("dashboardapp/finaldata/performance_SVM.csv", index=False)

### ----- cluster data SVM + average feature importance ----- ###

# make copy of original feature importance df of all targets
df_feat_imp_SVM_c = df_feat_imp_SVM.copy()

for target in targets_sorted:
    df_cluster_info = pd.read_csv('clusters info/SVM/SVM_{}.csv'.format(target))
    df_cluster_info.rename(columns={'Unnamed: 0': 'cluster number'}, inplace=True)
    df_cluster_info.set_index(df_cluster_info.columns[0], inplace=True)

    # drop target from list of feature names
    cols = data.columns.drop(target)

    # replace numbers with feature names and add empty column for iteration process later
    df_cluster_info.replace(to_replace=list(range(0, len(data.columns)-1)), value=cols, inplace=True)
    df_cluster_info['empty'] = np.nan

    # set row counter
    r = 0

    # iterate through list of features present in selected subset
    for i in df_cluster_info.iloc[:, 0]:

        # find corresponding feature importance
        imp = df_feat_imp_SVM_c.loc[df_feat_imp_SVM_c['Feature'] == i, target]

        # set column counter
        c = 1

        # for the features in each cluster
        while not pd.isna(df_cluster_info.iloc[r, c]):

            # extract the c-th feature in a particular cluster (r)
            feature = df_cluster_info.iloc[r, c]

            # check if feature is present in feature importance data
            try:
                row = df_feat_imp_SVM_c[df_feat_imp_SVM_c['Feature'] == feature].index[0]
                # print(row)
                df_feat_imp_SVM_c.at[row, target] = imp  # represents first target

            # if not, add it to the end of the df
            except:
                df_feat_imp_SVM_c.loc[len(df_feat_imp_SVM_c)] = {'Feature': feature, target: float(imp)}

            # move onto the next column (feature)
            c += 1
        # move onto the next row (feature present in selected subset)
        r += 1

# calculate average importance per feature and add it to the df
df_feat_imp_SVM_c['average'] = df_feat_imp_SVM_c.mean(axis=1, skipna=True, numeric_only=True)

# store data as csv
df_feat_imp_SVM_c.to_csv("dashboardapp/finaldata/complete_importance_SVM.csv", index=False)


### ----- feature importance MLP ----- ###

path_imp_MLP = str(os.getcwd())+'/feature importance/MLP/'
feat_imp_MLP = glob.glob(os.path.join(path_imp_MLP, '*.csv'))

df_feat_imp_MLP = multi_csv_to_df(feat_imp_MLP, axis=1, index_col='Unnamed: 0')
df_feat_imp_MLP = df_feat_imp_MLP.iloc[:, ::2]
df_feat_imp_MLP.columns = targets_sorted

df_feat_imp_MLP.insert(loc=0, column='Feature', value=df_feat_imp_MLP.index)
df_feat_imp_MLP.reset_index(drop=True, inplace=True)

# store data as csv
df_feat_imp_MLP.to_csv("dashboardapp/finaldata/importance_MLP.csv", index=True)


### ----- model performance MLP ----- ###

path_perf_MLP = str(os.getcwd())+'/parameter_search_outputs/MLP/'
perf_MLP = glob.glob(os.path.join(path_perf_MLP, '*.csv'))

df_perf_MLP = multi_csv_to_df(perf_MLP, axis=0, index_col=None)
df_perf_MLP.rename(columns={'Unnamed: 0': 'model'}, inplace=True)
df_perf_MLP = df_perf_MLP.loc[df_perf_MLP['model'] == 'best model'].reset_index(drop=True)
df_perf_MLP.index = targets_sorted
df_perf_MLP.loc['average'] = df_perf_MLP.mean(axis=0, numeric_only=True)


df_perf_MLP = df_perf_MLP.T
df_perf_MLP = df_perf_MLP.drop(['model', 'number units', 'L2 regularization',
                                'learning rate', 'beta 1', 'beta 2'], axis=0)

df_perf_MLP.insert(loc=0, column='Metric', value=df_perf_MLP.index)
df_perf_MLP.reset_index(drop=True, inplace=True)

# store data as csv
df_perf_MLP.to_csv("dashboardapp/finaldata/performance_MLP.csv", index=False)

### ----- cluster data MLP + average feature importance ----- ###

# make copy of original feature importance df of all targets
df_feat_imp_MLP_c = df_feat_imp_MLP.copy()

for target in targets_sorted:
    df_cluster_info = pd.read_csv('clusters info/MLP/MLP_{}.csv'.format(target))
    df_cluster_info.rename(columns={'Unnamed: 0': 'cluster number'}, inplace=True)
    df_cluster_info.set_index(df_cluster_info.columns[0], inplace=True)

    # drop target from list of feature names
    cols = data.columns.drop(target)

    # replace numbers with feature names and add empty column for iteration process later
    df_cluster_info.replace(to_replace=list(range(0, len(data.columns)-1)), value=cols, inplace=True)
    df_cluster_info['empty'] = np.nan

    # set row counter
    r = 0

    # iterate through list of features present in selected subset
    for i in df_cluster_info.iloc[:, 0]:

        # find corresponding feature importance
        imp = df_feat_imp_MLP_c.loc[df_feat_imp_MLP_c['Feature'] == i, target]

        # set column counter
        c = 1

        # for the features in each cluster
        while not pd.isna(df_cluster_info.iloc[r, c]):

            # extract the c-th feature in a particular cluster (r)
            feature = df_cluster_info.iloc[r, c]

            # check if feature is present in feature importance data
            try:
                row = df_feat_imp_MLP_c[df_feat_imp_MLP_c['Feature'] == feature].index[0]
                # print(row)
                df_feat_imp_MLP_c.at[row, target] = imp  # represents first target

            # if not, add it to the end of the df
            except:
                df_feat_imp_MLP_c.loc[len(df_feat_imp_MLP_c)] = {'Feature': feature, target: float(imp)}

            # move onto the next column (feature)
            c += 1
        # move onto the next row (feature present in selected subset)
        r += 1


# calculate average importance per feature and add it to the df
df_feat_imp_MLP_c['average'] = df_feat_imp_MLP_c.mean(axis=1, skipna=True, numeric_only=True)

# store data as csv
df_feat_imp_MLP_c.to_csv("dashboardapp/finaldata/complete_importance_MLP.csv", index=False)
