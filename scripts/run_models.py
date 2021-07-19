import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append('..')
from src.model_support import *
from src import *



# load models which were set as best models
with open('../results/best_models.pickle', 'rb') as handle:
    best_res = pickle.load(handle)

# load data
graz_all = pd.read_csv('../data/graz_all_p3.csv', index_col=0)
graz_all.index = pd.DatetimeIndex(graz_all.index)

#set vars
non_g_renamed = set(graz_all.columns).difference(set(gases))

#set lags
X = prep_lags(graz_all, non_g_renamed, temporal)

# set indices
train_rand = X[X.index < '2020'].index
test_rand = X[X.index > '2020'].index


predictive_datasets = {'X': X}
preproc_decision = False

# map name to algorithms
model_mapper = {'RandomForest':RandomForestRegressor}
algo_mapper = {'RandomForest':'rf_'}

# dataframe for stoing
all_results = pd.DataFrame()

for varname in gases:
    print('\n==================== %s =============   \n' % varname)


    predictive_set_key = 'X'

    # set y
    y = graz_all[varname]
    if varname.endswith('PM10'):
        y = y.drop(y[(graz_all.index >= '2020-03-26') & (graz_all.index <= '2020-03-29')].index, axis=0)
    y.name = varname
    y.dropna(inplace=True)


    X = SetXMatrix(predictive_datasets, predictive_set_key, preprocess=preproc_decision)
    X, y = index_fixer(X, y)


    X_train, y_train = X.loc[train_rand], y.loc[train_rand]
    X_ext, y_ext = X.loc[test_rand], y.loc[test_rand]
    y_ext.name = varname + '_TRUE'

    results_collector_dict = {}

    #load parameters, features
    model_params = best_res[varname]['params']
    model_features = best_res[varname]['features']
    print(len(model_features))
    model_alg = best_res[varname]['algo']

    clean_params = ret_params(model_params, algo_mapper[model_alg])
    model = model_mapper[model_alg](**clean_params)
    model.fit(X_train[model_features], y_train)

    y_predicted = pd.Series(model.predict(X_ext[model_features]),
                            index=y_ext.index,
                            name=varname + '_PRED')

    if varname == 'no':
        all_results = pd.DataFrame(y_ext)
        all_results.drop(y_ext.name, axis=1, inplace=True)
    all_results = pd.concat([all_results, y_predicted, y_ext], axis=1)
    print(y_predicted.head(1))

#store results
print(all_results.head(1))
all_results.to_csv('../results/all_results.csv')


