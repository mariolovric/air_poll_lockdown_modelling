# Modelling environment concentrations of pollutants COVID
Please cite this paper:

Lovrić M, Pavlović K, Vuković M, et al (2020) Understanding the true effects of the COVID-19 lockdown on air pollution by means of machine learning.
Environ Pollut 115900. https://doi.org/10.1016/j.envpol.2020.115900


## Project structure

    .
    ├── data                 # Data for modelling
    ├── results              # Results 
    ├── scripts              # Automated tests
    ├── src                  # Source, models, tools, utilities
    ├── LICENSE
    └── README.md            # Brief repo description and installation recommendation
______________________________________________


## Best regressor

This scripts creates a pickle file with model parameters and results.
Should be run as:
>> python train_regressors.py
 

## Running the winning models

>>> python run_models.py

______________________________________________
The code is set up as follows:

> `src` has all the modules necessary for modelling

> `src/configs.py` 		   | Parameter space definitions for ML models
> `src/models.py`  		   | Optimization and modelling modules
> `src/model_support.py`   | Preprocessing routines and aux files

______________________________________________
Installation recommendation:

It is recommended to create a new conda environment. Conda should be pre-installed.
In the conda terminal run following commands:
`conda create -n env python=3.6 scikit-learn=0.22 numpy pandas jupyter`
`conda install -c conda-forge imbalanced-learn bayesian-optimization eli5 ` 
