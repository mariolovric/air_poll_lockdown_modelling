param_space = {
    'rf_':
    {
        'max_depth': (8, 50),
        'n_estimators': (100, 500),
        'max_samples': (.3, .5),
        'min_samples_split': (3, 6)
    },

    'lg_':
    {
        'num_leaves': (20, 250),
        'max_depth': (2, 50),
        'lambda_l2': (0.0, .1),
        'lambda_l1': (0.0, .1),
        'min_data_in_leaf': (2, 10)
    },
    'pls_':
    {
        'n_components': (2, 10),

    },
    'lasso_':
    {
        'alpha': (0.01, 1000)
    },

}


