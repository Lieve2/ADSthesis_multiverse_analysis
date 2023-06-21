import time
import pandas as pd
from pipeline_RF import KfoldCV
from utility_functions import encode_scale_data_perm


## ------- define processing pipeline function --------- ##

def processing_pipeline_RF(data, targets, threshold, filename, num_feat):
    """
            Overarching function for running the entire processing pipeline

            :param data:        cleaned data set with missing values
            :param targets:     list of targets (list(str))
            :param threshold:   the (cut) level for clustering
            :param filename:    name of the file (str)
            :param num_feat:    list of numerical features in the data (list(str))
    """

    for target in targets:

        X, y, features, clusters = encode_scale_data_perm(data, target, threshold, num_feat)

        df_clusters = pd.DataFrame.from_dict(clusters, orient='index')

        df_clusters.to_csv('clusters info/RF/{}_{}.csv'.format(filename, target))

        # timer - start
        t = time.process_time()

        # perform k-fold CV
        KfoldCV(3, X, y,
                filename='{}_{}'.format(filename, target),
                features=features)

        # timer - end
        elapsed_time = time.process_time() - t
        print(f"Elapsed time for K-fold CV {target}: {elapsed_time} sec")


## ------- run modeling pipeline --------- ##


# load data
data = pd.read_csv('ESS8 data/ESS8_subset_cleaned_timeadj_wmissingvals.csv', low_memory=False)

# make df with missingness percentage of features with missingness
missing_cols = data.columns[data.isnull().any()].to_list()
percent_missing = data[missing_cols].isnull().sum() * 100 / len(data)
missing_info = pd.DataFrame({'column name': missing_cols,
                             'percentage missing': percent_missing})

# extract rows with > 5% missing (75 features) and set as targets
many_missing = missing_info[missing_info['percentage missing'] > 5]
targets = many_missing['column name'].tolist()

# define numerical features
num_feat = ['dweight', 'pspwght', 'pweight', 'anweight', 'nwspol',
            'netustm', 'agea', 'eduyrs', 'wkhct', 'wkhtot',
            'wkhtotp', 'inwtm']

# start the modeling pipeline
processing_pipeline_RF(data=data, targets=targets, num_feat=num_feat, threshold=1, filename='RF')
