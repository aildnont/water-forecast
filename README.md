# Water Demand Forecasting
![alt text](img/readme/london_logo.png "The City of London, Canada")

THe purpose of this project is to deliver a machine learning solution to
forecasting aggregate water demand. This work was led by the Artificial
Intelligence Research and Development Lab out of the Information
Technology Services division. This repository contains the code used to
fit multiple types of machine learning models to predict future daily
water consumption. This repository is intended to serve as an example
for other municipalities who wish to explore water demand forecasting in
their own locales.

## Table of Contents
1. [**_Getting Started_**](#getting-started)
2. [**_Use Cases_**](#use-cases)  
   i)
   [_Train a model and visualize forecasts_](#train-a-model-and-visualize-a-forecast)  
   ii)
   [_Bayesian hyperparameter optimization_](#bayesian-hyperparameter-optimization)  
   iii)
   [_Batch predictions from raw data_](#batch-predictions-from-raw-data)  
   iv) [_Cross validation_](#cross-validation)  
   v) [_Exclusion of features_](#exclusion-of-features)  
   vi)
   [_Client clustering experiment (using K-Prototypes)_](#client-clustering-experiment-using-k-prototypes)
3. [**_Project Structure_**](#project-structure)
4. [**_Project Config_**](#project-config)
5. [**_Contact_**](#contact)

## Getting Started
1. Clone this repository (for help see this
   [tutorial](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)).
2. Install the necessary dependencies (listed in
   [requirements.txt](requirements.txt)). To do this, open a terminal in
   the root directory of the project and run the following:
   ```
   $ pip install -r requirements.txt
   ```
3. Obtain raw water consumption data and preprocess it accordingly. The
   data that was used for our experiments was organized in a schema
   specific to London, Canada. Your preprocessed dataset should be a CSV
   file with at least the following columns:
   - _Date_: A timestamp representing a date
   - _Consumption_: A float representing an estimate of total water
     consumption on the date

   Any other features you wish to use must have their own column. The
   model assumes all datatypes are numerical. As a result, categorical
   and Boolean features must be converted to a numeric representation.
4. Update the _TRAIN >> MODEL_ field of [_config.yml_](config.yml) with
   the appropriate string representing the model type you wish to
   train. To train a model, ensure the _TRAIN >>
   EXPERIMENT_ field is set to _'train_single'_.
5. Execute [_train.py_](src/train.py) to train your chosen model on your
   preprocessed data. The trained model will be serialized within
   _results/models/_, and its filename will resemble the following
   structure: _{modeltype}{yyyymmdd-hhmmss}.{ext}_, where _{modeltype}_
   os the type of model trained, _{yyyymmdd-hhmmss}_ is the current
   time, and _{ext}_ is the appropriate file extension.
6. Navigate to _results/experiments/_ to see the performance metrics
   achieved by the model's forecast on the test set. The file name will
   be _{modeltype}_eval_{yyyymmdd-hhmmss}.csv_. You can find a
   visualization of the test set forecast in
   _img/forecast_visualizations/_. Its filename will be
   _{modeltype}_forecast_{yyyymmdd-hhmmss}.png_.


## Use Cases

### Train a model and visualize a forecast
1. Once you have obtained a preprocessed dataset (see step 3 of
   [_Getting Started_](#getting-started), ensure that the preprocessed
   dataset is located in _data/preprocessed/_ with filename
   _"preprocessed_data.csv"_.
2. In [config.yml](config.yml), set _EXPERIMENT >> TRAIN_ to
   _'single_train'_. Set _TRAIN >> MODEL_ to the appropriate string
   representing the model type you wish to train.
3. Execute [_train.py_](src/train.py) to train your chosen model on your
   preprocessed data. The trained model will be serialized within
   _results/models/_, and its filename will resemble the following
   structure: _{modeltype}{yyyymmdd-hhmmss}.{ext}_, where _{modeltype}_
   os the type of model trained, _{yyyymmdd-hhmmss}_ is the current
   time, and _{ext}_ is the appropriate file extension.
4. Navigate to _results/experiments/_ to see the performance metrics
   achieved by the model's forecast on the test set. The file name will
   be _{modeltype}_eval_{yyyymmdd-hhmmss}.csv_. You can find a
   visualization of the test set forecast in
   _img/forecast_visualizations/_. Its filename will be
   _{modeltype}_forecast_{yyyymmdd-hhmmss}.png_. This visualization
   displays training set predictions, test set forecasts, test set
   predictions (depending on the model type), training residuals, and
   test error. The image below is an example of one of a test set
   forecast visualization.

![alt text](img/readme/test_set_forecast_visualization.png "Test set
forecast visualization")

### Bayesian hyperparameter optimization
Hyperparameter optimization is an important part of the standard machine
learning workflow. We chose to conduct Bayesian hyperparameter
optimization. The results of the optimization informed the final
hyperparameters currently set in the _HPARAMS_ sections of
[config.yml](config.yml). The objective of the Bayesian optimization was
the minimization of mean absolute error in the average test set forecast
resulting from a time series cross validation. With the help of the
[scikit-optimize](https://scikit-optimize.github.io/stable/index.html)
package, we were able to visualize the effects of single hyperparameters
and pairs of hyperparameters on the objective.

To conduct your own Bayesian hyperparameter optimization, you may follow
the steps below. Note that if you are not planning on changing the
default hyperparameter ranges set in the _HPARAM_SEARCH_ section of
[config.yml](config.yml), you may skip step 2.
1. In [config.yml](config.yml), set _EXPERIMENT >> TRAIN_ to
   _'hparam_search'_. Set _TRAIN >> MODEL_ to the appropriate string
   representing the model type you wish to train.
2. In the _TRAIN >> HPARAM_SEARCH >> MAX_EVALS_ field of
   [config.yml](config.yml), set the maximum number of combinations of
   hyperparameters to test. In the _TRAIN >> HPARAM_SEARCH >>
   LAST_FOLDS_ field, set the number of folds from cross validation to
   average the test set forecast objective over. If you wish to change
   the objective metric, update _TRAIN >> HPARAM_SEARCH >>
   HPARAM_OBJECTIVE_ accordingly.
3. Set the ranges of hyperparameters you wish to study in the
   _HPARAM_SEARCH >> {MODEL}_ subsection of [config.yml](config.yml),
   where _{{MODEL}}_ is the model name you set in Step 1. The config
   file already has a comprehensive set of hyperparameter ranges defined
   for each model type, so you may not need to change anything in this
   step. Consider whether your hyperparameter ranges are sets, integer
   ranges, or float ranges, and whether they need to be investigated on
   a uniform or logarithmic range. See below for an example of how to
   correctly set ranges for different types of hyperparameters.
   ```
    FC0_UNITS:
      TYPE: 'set'
      RANGE: [32, 64, 128]
    DROPOUT:
      TYPE: 'float_uniform'
      RANGE: [0.0, 0.5]
    LR:
      TYPE: 'float_log'
      RANGE: [0.00001, 0.001]   
    PATIENCE:
      TYPE: 'int_uniform'
      RANGE: [5, 15]      
   ```
4. Execute [train.py](src/train.py). A CSV log detailing the trials in
   the Bayesian optimization will be located in
   _results/experiments/hparam_search\_{modelname}\_{yyyymmdd-hhmmss}.csv_,
   where _{modeltype}_ is the type of model investigated, and
   _{yyyymmdd-hhmmss}_ is the current time. Each row in the CSV will
   detail the hyperparameters used for a particular trial, along with
   the value of the objective from the cross validation run with those
   hyperparameters. The final row details the best trial, which contains
   the optimal hyperparameter values. The visualization of the search
   can be found at
   _img/experiment_visualizations/Bayesian_opt\_{modeltype}\_{yyyymmdd-hhmmss}.png_.
   See below for an example of a Bayesian hyperparameter optimization
   visualization for an LSTM-based model:

![alt text](img/readme/hparam_visualization.png "A sample Bayesian
hyperparameter optimization visualization")

### Batch predictions from raw data

### Cross validation

### Exclusion of features

### Client clustering experiment (using K-Prototypes)

## Project Structure
The project looks similar to the directory structure below. Disregard
any _.gitkeep_ files, as their only purpose is to force Git to track
empty directories. Disregard any _.\__init\__.py_ files, as they are
empty files that enable Python to recognize certain directories as
packages.

```
├── data
│   ├── processed                 <- Products of preprocessing
│   ├── raw                       <- Raw data
│   └── serializations            <- Serialized sklearn transformers
|
├── img
|   ├── data_visualizations       <- Visualizations of preprocessed data, clusterings
|   ├── experiment_visualizations <- Visualizations for experiments
|   ├── forecast_visualizations   <- Visualizations of model forecasts
|   └── readme                    <- Image assets for README.md
├── results
│   ├── experiments               <- Experiment results
│   ├── logs                      <- TensorBoard logs
│   ├── models                    <- Trained model serializations
│   └── predictions               <- Model predictions
|
├── src
│   ├── data
|   |   ├── data_merge.py         <- Script for merging new raw data with existing raw data
|   |   ├── kprototypes.py        <- Script for learning client clusters
|   |   └── preprocess.py         <- Data preprocessing script
│   ├── models                    <- TensorFlow model definitions
|   |   ├── arima.py              <- Script containing ARIMA model class definition
|   |   ├── model.py              <- Script containing abstract model class definition
|   |   ├── nn.py                 <- Script containing neural network model class definitions
|   |   ├── prophet.py            <- Script containing Prophet model class definition
|   |   ├── sarima.py             <- Script containing SARIMA model class definition
|   |   └── skmodels.py           <- Script containing scikit-learn model class definitions
|   ├── visualization             <- Visualization scripts
|   |   └── visualize.py          <- Script for visualization production
|   ├── predict.py                <- Script for prediction on raw data using trained models
|   └── train.py                  <- Script for training experiments
|
├── .gitignore                    <- Files to be be ignored by git.
├── config.yml                    <- Values of several constants used throughout project
├── config_private.yml            <- Private information, e.g. database keys (not included in repo)
├── LICENSE                       <- Project license
├── README.md                     <- Project description
└── requirements.txt              <- Lists all dependencies and their respective versions
```

## Project Config
This project contains several configurable variables that are defined in
the project config file: [config.yml](config.yml). When loaded into
Python scripts, the contents of this file become a dictionary through
which the developer can easily access its members.

For user convenience, the config file is organized into major components
of the model development pipeline. Many fields need not be modified by
the typical user, but others may be modified to suit the user's specific
goals. A summary of the major configurable elements in this file is
below.

## Contact

**Matt Ross**  
Manager, Artificial Intelligence  
Information Technology Services, City Manager’s Office  
City of London  
Suite 300 - 201 Queens Ave, London, ON. N6A 1J1  
maross@london.ca

**Blake VanBerlo**  
Data Scientist  
City of London Research and Development Lab  
C: vanberloblake@gmail.com