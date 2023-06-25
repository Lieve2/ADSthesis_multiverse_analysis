import glob
import os.path
import numpy as np
import pandas
import pandas as pd
from utility_functions import multi_csv_to_df

pandas.set_option('display.max_rows', 188)

# load data
data = pd.read_csv('ESS8 data/ESS8_subset_cleaned_timeadj_wmissingvals.csv', low_memory=False)

# make df with missingness percentage of features with missingness
missing_cols = data.columns[data.isnull().any()].to_list()
percent_missing = data[missing_cols].isnull().sum() * 100 / len(data)
missing_info = pd.DataFrame({'column name': missing_cols,
                             'percentage missing': percent_missing})

# extract rows with > 5% missing (74 features)
many_missing = missing_info[missing_info['percentage missing'] > 5]
targets = many_missing['column name'].tolist()[0:64] # currently only the first 64 targets are used
targets_sorted = sorted(targets)
print(targets_sorted)

### ----- feature importance MLP ----- ###

path_imp_RFMI = str(os.getcwd())+'/feature importance/RF/RF_MI/'
feat_imp_RFMI = glob.glob(os.path.join(path_imp_RFMI, '*.csv'))

df_feat_imp_RFMI = multi_csv_to_df(feat_imp_RFMI, axis=1, index_col='Unnamed: 0')
df_feat_imp_RFMI = df_feat_imp_RFMI.iloc[:, ::2]
df_feat_imp_RFMI.columns = targets_sorted

df_feat_imp_RFMI.insert(loc=0, column='Feature', value=df_feat_imp_RFMI.index)
df_feat_imp_RFMI.reset_index(drop=True, inplace=True)

print(df_feat_imp_RFMI)

# store data as csv
df_feat_imp_RFMI.to_csv("dashboardapp/finaldata/importance_RFMI.csv", index=True)


### ----- model performance MLP ----- ###

path_perf_RFMI = str(os.getcwd())+'/parameter_search_outputs/RF/RF_MI/'
perf_RFMI = glob.glob(os.path.join(path_perf_RFMI, '*.csv'))

df_perf_RFMI = multi_csv_to_df(perf_RFMI, axis=0, index_col=None)
df_perf_RFMI.rename(columns={'Unnamed: 0': 'model'}, inplace=True)
df_perf_RFMI = df_perf_RFMI.loc[df_perf_RFMI['model'] == 'best model'].reset_index(drop=True)
df_perf_RFMI.index = targets_sorted
df_perf_RFMI.loc['average'] = df_perf_RFMI.mean(axis=0, numeric_only=True)


df_perf_RFMI = df_perf_RFMI.T
df_perf_RFMI = df_perf_RFMI.drop(['model', 'number estimators',
                              'max features', 'max depth',
                              'min split', 'min leaf'], axis=0)

df_perf_RFMI.insert(loc=0, column='Metric', value=df_perf_RFMI.index)
df_perf_RFMI.reset_index(drop=True, inplace=True)

print(df_perf_RFMI)

# store data as csv
df_perf_RFMI.to_csv("dashboardapp/finaldata/performance_RFMI.csv", index=False)

### ----- cluster data MLP + average feature importance ----- ###

# make copy of original feature importance df of all targets
df_feat_imp_RFMI_c = df_feat_imp_RFMI.copy()

for target in targets_sorted:
    df_cluster_info = pd.read_csv('clusters info/RF/RF_MI/RF_mi_{}.csv'.format(target))
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
        imp = df_feat_imp_RFMI_c.loc[df_feat_imp_RFMI_c['Feature'] == i, target]

        # set column counter
        c = 1

        # for the features in each cluster
        while not pd.isna(df_cluster_info.iloc[r, c]):

            # extract the c-th feature in a particular cluster (r)
            feature = df_cluster_info.iloc[r, c]

            # check if feature is present in feature importance data
            try:
                row = df_feat_imp_RFMI_c[df_feat_imp_RFMI_c['Feature'] == feature].index[0]
                # print(row)
                df_feat_imp_RFMI_c.at[row, target] = imp  # represents first target

            # if not, add it to the end of the df
            except:
                df_feat_imp_RFMI_c.loc[len(df_feat_imp_RFMI_c)] = {'Feature': feature, target: float(imp)}

            # move onto the next column (feature)
            c += 1
        # move onto the next row (feature present in selected subset)
        r += 1


# calculate average importance per feature and add it to the df
df_feat_imp_RFMI_c['average'] = df_feat_imp_RFMI_c.mean(axis=1, skipna=True, numeric_only=True)

print(df_feat_imp_RFMI_c[['Feature', 'average']].sort_values('average', ascending=False)[0:10])

# store data as csv
df_feat_imp_RFMI_c.to_csv("dashboardapp/finaldata/complete_importance_RFMI.csv", index=False)

