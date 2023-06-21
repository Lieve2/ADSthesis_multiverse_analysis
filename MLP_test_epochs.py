import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
from utility_functions import encode_scale_data_perm


def epoch_test(data, targets, threshold, num_feat, classifier):
    """
            :param data:        cleaned data set with missing values
            :param targets:     list of targets (list(str))
            :param threshold:   the (cut) level for clustering
            :param num_feat:    list of numerical features in the data (list(str))
            :param classifier:  classification model for which to test the epochs-performance distribution
    """

    # clean the data and split into train, validation and test sets
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

        # plot the results
        plt.plot(range(len(train_loss_)), train_loss_, label="train loss")
        plt.plot(range(len(valid_loss_)), valid_loss_, label="validation loss")
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig('figures/justification_epochs_{}.png'.format(target), dpi=300, bbox_inches='tight')
        plt.show()


### ----- define the input variables and run the test ----- ###

data3 = pd.read_csv('ESS8 data/ESS8_subset_cleaned_timeadj_wmissingvals.csv', low_memory=False)

# make df with missingness percentage of features with missingness
missing_cols = data3.columns[data3.isnull().any()].to_list()
percent_missing = data3[missing_cols].isnull().sum() * 100 / len(data3)
missing_info = pd.DataFrame({'column name': missing_cols,
                             'percentage missing': percent_missing})

# extract rows with > 5% missing (75 features)
many_missing = missing_info[missing_info['percentage missing'] > 5]

targets3 = many_missing['column name'].tolist()

num_feat = ['dweight', 'pspwght', 'pweight', 'anweight', 'nwspol',
            'netustm', 'agea', 'eduyrs', 'wkhct', 'wkhtot',
            'wkhtotp', 'inwtm']

classifier = MLPClassifier(hidden_layer_sizes=(100, 50, 25, 13),  # hidden_layers
                           activation='logistic',  # sigmoid
                           solver='adam',
                           alpha=0.01,  # regularization (maybe as second par?)
                           batch_size=2048,
                           learning_rate_init=0.001,
                           max_iter=500,  # nr epochs
                           shuffle=True,
                           random_state=144,
                           tol=0.0001,  # optim tolerance (default used)
                           warm_start=True,
                           early_stopping=False,  # set to true if want to validate (10%) within
                           beta_1=0.95,
                           beta_2=0.95,
                           n_iter_no_change=20  # default nr epochs for tol
                           )

epoch_test(data=data3, targets=['occf14b'], num_feat=num_feat, threshold=1, classifier=classifier)
