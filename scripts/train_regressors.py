import sys

sys.path.append('..')
from src.model_support import *
from src import *
from src.models import RegressorTest

# load data
graz_all = pd.read_csv('../data/graz_all_p3.csv', index_col=0)
graz_all.index = pd.DatetimeIndex(graz_all.index)

#set variables for modelling
non_g_renamed = set(graz_all.columns).difference(set(gases))

#set lags (time based variables)
X = prep_lags(graz_all, non_g_renamed, temporal)

#split train test
train_rand = X[X.index < '2020'].index
test_rand = X[X.index > '2020'].index

#historic reasons, can be removed
predictive_datasets = {'X': X}
preproc_decision = False

for varname in gases:

    try:

        predictive_set_key = 'X'

        # set y
        y = graz_all[varname]

        #exception for PM10, exclusion of the saharan dust event
        if varname.endswith('PM10'):
            y = y.drop(y[(graz_all.index >= '2020-03-26') & (graz_all.index <= '2020-03-29')].index, axis=0)
        y.name = varname


        X = SetXMatrix(predictive_datasets, predictive_set_key, preprocess=preproc_decision)
        X, y = index_fixer(X, y)


        X_train, y_train = X.loc[train_rand], y.loc[train_rand]
        X_ext, y_ext = X.loc[test_rand], y.loc[test_rand]

        #dictionary for results collection
        results_collector_dict = {}

        # Random Forest, params can be changes
        test_rf = RegressorTest(2, 10, 5, 'rf_', X_train, y_train)
        results_collector_dict.update({'RandomForest': test_rf})

        # storing results
        with open('../results/regressor_result_key-%s_test_preprocess-%s_var-%s.pickle' % (predictive_set_key,
            preproc_decision, varname), 'wb') as handle:
            pickle.dump(results_collector_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except:
        print('error ', varname)




