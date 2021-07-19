from src import *
from src.configs import param_space
from src.model_support import *
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance

def rmse(y_actual, y_predicted):
    rms = sqrt(mean_squared_error(y_actual, y_predicted))

    return rms

def rmse_err(y, y_pred):
    '''
    :param y: true y
    :param y_pred: predicted y
    :return: RMSE in %
    '''
    n = len(y)
    del_y = np.divide(np.subtract(y_pred, y), y)
    rmse_err = np.sqrt(np.sum(np.square(del_y)) / n) * 100
    return rmse_err

scor = make_scorer(rmse, greater_is_better=False)

def RegressorTest(runs, iters, inits, test, X_train, y_train):
    """

    :param runs: number of runs
    :param iters: iterations in BayesOpt
    :param inits: initial points of parameter space in BayesOpt
    :param test: algorithm, can be: rf_ , lg_ , pls_ , lasso_
    :param X_train: train set
    :param y_train: train set
    :return:
    """

    assert X_train.isnull().sum().any()==False, 'There are NaNs'


    results_dictionary = {}
    features_dictionary = {}
    parameters_dictionary = {}
    final_result = {}

    for i in range(runs):

        try:
            print('\n===== RUN:', i, '=====', )

            if i == 0:
                # run 0, first time regression, no feat sel
                regressor_instance = RegressorEvaluatorModule(X_train, y_train)
                features_list = X_train.columns

            else:
                # iterative feature sel, if run is not 0
                print('**** started feature selection')
                features_list = regressor_instance.feat_selector_module(best_model_from_BO)
                #update
                features_dictionary.update({test + str(i): features_list})
                print('len features_list:', len(features_list))
                # train with feature selection
                regressor_instance = RegressorEvaluatorModule(X_train[features_list], y_train)

            regressor_object = {'rf_': regressor_instance.rf_eval}

            # call optimizer, special case with pls because n_components can be lower than len(features)
            try:
                print(test, param_space[test])
                BO = BayesianOptimization(regressor_object[test], param_space[test])
                BO.maximize(n_iter=iters, init_points=inits)
            except:
                print('error BO')

            #extract best params from the bayesian optimizer
            best_params_ = BO.max["params"]
            print('\n** Results from run **', )

            #return model w best params
            best_model_from_BO = regressor_object[test](**best_params_, return_model=True,
                                print_res=True, exp_=test + str(i))
            #if first run, lasso special case for coefficients
            if i == 0:
                features_dictionary.update({test + str(i): features_list})

            # store results_collector_dict
            results_dictionary.update(regressor_instance.results_dictionary)
            parameters_dictionary.update({test + str(i): best_params_})

        except:
            print('Error in RegressorTest')

    final_result.update({test: results_dictionary, 'features': features_dictionary, 'params': parameters_dictionary})
    return final_result


class RegressorEvaluatorModule:
    '''
    Evalutation of 4 regressors packed in modules: lasso_eval, pls_eval, rf_eval, lgbm_eval

    '''

    def __init__(self, X_train, y_train):
        """
        :param X_train:
        :param y_train:

        """

        assert X_train.shape[0] != 0, 'One dim is zero'
        assert X_train.shape[1] != 0, 'One dim is zero'

        self.X_train = X_train
        self.y_train = y_train
        self.results_dictionary = {}


    def rf_eval(self, max_depth, n_estimators, min_samples_split, max_samples,
                return_model=False, print_res=False, exp_='rf'):
        '''
        Evaluator RF Regression, arguments are algorithm specific
        :param max_depth:
        :param n_estimators:
        :param min_samples_split:
        :param max_samples:
        :param return_model:
        :param print_res:
        :param exp_:
        :return:
        '''
        params = {'bootstrap': True, 'max_depth': int(max_depth), 'n_estimators': int(n_estimators),
                  'max_samples': float(max_samples), 'min_samples_split': int(min_samples_split),
                  'random_state': 42, 'n_jobs':20}

        model = RandomForestRegressor(**params)
        model.fit(self.X_train, self.y_train)

        validation_score = self.internal_validator(model, print_res)

        if return_model:
            self.results_dictionary.update({str(exp_): validation_score})
            return model

        # We optmize based on MSE of val
        return validation_score


    def internal_validator(self, model, print_res=False):
        '''
        Internal validator used in the models.
        :param model: pass model from xxx_eval
        :param print_res: if True, will print the results_collector_dict
        :return: validation score and results_collector_dict dictionary
        '''

        cvs = cross_val_score(estimator=model,
                              X=self.X_train,
                              y=self.y_train,
                              scoring=scor, cv=10).mean()

        return cvs

    def feat_selector_module(self, model):
        result = permutation_importance(model, self.X_train, self.y_train, n_repeats=3,
                                        random_state=42, n_jobs=24)

        weights: pd.Series = pd.Series(result['importances_mean'], index=self.X_train.columns). \
            sort_values(ascending=False)

        #TODO: changed here to length of instances, bilo weights
        one_third_of_feature_count = int(len(self.X_train) / 3)
        selected_features_list = weights[weights > .0001].index.tolist()

        if len(selected_features_list) == 0:
            # if there are no pos.weights
            print('error in selection', len(self.X_train.columns.tolist()), ', or all features selected')
            return self.X_train.columns.tolist()

        elif len(selected_features_list) > one_third_of_feature_count:
            # reduce to one third
            return selected_features_list[0:one_third_of_feature_count]

        else:
            return selected_features_list



