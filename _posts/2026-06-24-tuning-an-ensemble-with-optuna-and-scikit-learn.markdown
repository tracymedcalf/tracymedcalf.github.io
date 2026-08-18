---
layout: post
title: "Tuning an Ensemble with Optuna and Scikit-Learn"
date: "2026-06-24"
updated: "2026-06-24"
categories:
  - "machine learning"
  - "scikit-learn"
  - "optuna"
  - "ensemble"
  - "decision tree"
  - "hyperparameter optimization"
coverImage: "/assets/images/verdant-stylized-forest.png"
excerpt: An unorthodox use of Optuna.
---

# Tuning an Ensemble Model with Optuna and Scikit-Learn

I recently became aware of a hyperparameter tuning library that is agnostic to the ML framework, so I decided to try it out by using it to tune an ensemble model with Scikit-Learn.

To be clear, this is an unorthodox use of Optuna. I was simply inspired by the success of ensemble libraries like Autogluon, which seem to make the very most of every scrap.


```python
%pip install numpy scikit-learn optuna plotly nbformat
```

```python
from sklearn import datasets
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import optuna
import scipy
import sklearn

# Set the random seed for reproducibility
random_seed = 42
rng = np.random.default_rng(random_seed)
```

    /home/mtrac/repos/blog-repos/tuning-an-ensemble-w-optuna/.venv/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


A very brief description of the Forest Cover Type dataset:

| Classes | Samples total | Dimensionality | Features |
| --- | --- | --- | --- |
| 7 | 581012 | 54 | int |


```python
# Fetch the forest cover type dataset
X, y = datasets.fetch_covtype(return_X_y=True, random_state=random_seed)
```

Let's examine the classes of the target variable, both for our own edification and for fitting the decision tree later on.


```python
# Display all classes
classes = np.unique(y)
classes
```




    array([1, 2, 3, 4, 5, 6, 7], dtype=int32)




```python
# Use a test size of 0.5 (half the dataset) because the dataset is enormous and I want to save time on training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_seed)
```

We set `splitter` to `random`.

The alternative, `best`, does an exhaustive search. A random splitter is more efficient, and that is attractive to me as I am someone who iterates on his models rapidly. A random splitter also makes more sense with the use of these trees in an ensemble, as each tree is thus randomly different.


```python
estimators = []
accuracies = []

def objective(trial):

    # Optuna estimates the best values for these hyperparameters
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0, 0.5)
    min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0, 1) 
    
    # Instantiate and fit the model
    model = DecisionTreeClassifier(
        criterion=criterion,
        splitter='random',
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        min_impurity_decrease=min_impurity_decrease,
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = model.score(X_val, y_val)
    accuracies.append(accuracy)
    
    estimators.append(model)

    return accuracy

dt_study = optuna.create_study(direction='maximize')
dt_study.optimize(objective, n_trials=100)
```

    [I 2026-06-24 19:22:52,050] A new study created in memory with name: no-name-ccf7ff8d-3919-4715-8f9a-1a78666d0992
    [I 2026-06-24 19:22:52,353] Trial 0 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 12, 'min_samples_split': 14, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.35291372414839983, 'min_impurity_decrease': 0.9800722957640594}. Best is trial 0 with value: 0.4878982810180504.
    [I 2026-06-24 19:22:52,623] Trial 1 finished with value: 0.4878982810180504 and parameters: {'criterion': 'entropy', 'max_depth': 14, 'min_samples_split': 19, 'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0.07945984754768459, 'min_impurity_decrease': 0.5381902880090894}. Best is trial 0 with value: 0.4878982810180504.
    [I 2026-06-24 19:22:52,890] Trial 2 finished with value: 0.4878982810180504 and parameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 7, 'min_weight_fraction_leaf': 0.048248010560859034, 'min_impurity_decrease': 0.4383341599564359}. Best is trial 0 with value: 0.4878982810180504.
    [I 2026-06-24 19:22:53,153] Trial 3 finished with value: 0.4878982810180504 and parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 5, 'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0.34678882890209656, 'min_impurity_decrease': 0.356381997431063}. Best is trial 0 with value: 0.4878982810180504.
    [I 2026-06-24 19:22:53,847] Trial 4 finished with value: 0.5202448312213592 and parameters: {'criterion': 'log_loss', 'max_depth': 12, 'min_samples_split': 16, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.03889220175463409, 'min_impurity_decrease': 0.08133959912654298}. Best is trial 4 with value: 0.5202448312213592.
    [I 2026-06-24 19:22:54,111] Trial 5 finished with value: 0.4878982810180504 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0.09178770236905243, 'min_impurity_decrease': 0.17554861288854706}. Best is trial 4 with value: 0.5202448312213592.
    [I 2026-06-24 19:22:54,387] Trial 6 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 11, 'min_samples_split': 13, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.47586366559211285, 'min_impurity_decrease': 0.3586881890646264}. Best is trial 4 with value: 0.5202448312213592.
    [I 2026-06-24 19:22:54,927] Trial 7 finished with value: 0.5613799186764484 and parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 4, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.14097791080200434, 'min_impurity_decrease': 0.02414647730758379}. Best is trial 7 with value: 0.5613799186764484.
    [I 2026-06-24 19:22:55,201] Trial 8 finished with value: 0.4878982810180504 and parameters: {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 18, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.35279107345386057, 'min_impurity_decrease': 0.18617875305706744}. Best is trial 7 with value: 0.5613799186764484.
    [I 2026-06-24 19:22:55,470] Trial 9 finished with value: 0.4878982810180504 and parameters: {'criterion': 'gini', 'max_depth': 15, 'min_samples_split': 13, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.2785296885822845, 'min_impurity_decrease': 0.4326233013984704}. Best is trial 7 with value: 0.5613799186764484.
    [I 2026-06-24 19:22:55,748] Trial 10 finished with value: 0.4878982810180504 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 10, 'min_weight_fraction_leaf': 0.19205611015874127, 'min_impurity_decrease': 0.8423326574112502}. Best is trial 7 with value: 0.5613799186764484.
    [I 2026-06-24 19:22:56,659] Trial 11 finished with value: 0.6655192444224521 and parameters: {'criterion': 'log_loss', 'max_depth': 9, 'min_samples_split': 9, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.004797471817953661, 'min_impurity_decrease': 0.026590408032416774}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:22:57,259] Trial 12 finished with value: 0.49189991609474837 and parameters: {'criterion': 'log_loss', 'max_depth': 8, 'min_samples_split': 8, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.1785495447149202, 'min_impurity_decrease': 0.00011370454089563997}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:22:57,764] Trial 13 finished with value: 0.5202448312213592 and parameters: {'criterion': 'log_loss', 'max_depth': 9, 'min_samples_split': 9, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0011358064776235965, 'min_impurity_decrease': 0.18209992621551763}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:22:58,046] Trial 14 finished with value: 0.4878982810180504 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 2, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.1473610719618433, 'min_impurity_decrease': 0.060087159319305514}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:22:58,329] Trial 15 finished with value: 0.4878982810180504 and parameters: {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 6, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.13860922478180265, 'min_impurity_decrease': 0.26011012853836596}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:22:58,839] Trial 16 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 9, 'min_samples_split': 7, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.22008532373450373, 'min_impurity_decrease': 0.0028365871887749694}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:22:59,554] Trial 17 finished with value: 0.6436393365030872 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 11, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 1.274715232924082e-05, 'min_impurity_decrease': 0.1237979388323397}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:22:59,824] Trial 18 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 11, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0019212020298222392, 'min_impurity_decrease': 0.5637108714224948}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:00,336] Trial 19 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 11, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.08758060977761152, 'min_impurity_decrease': 0.2667328965909744}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:00,858] Trial 20 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 13, 'min_samples_split': 15, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.044721658597066394, 'min_impurity_decrease': 0.12738383269171152}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:01,496] Trial 21 finished with value: 0.5671779867042448 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 4, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.13057886647685243, 'min_impurity_decrease': 0.0870722780609437}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:01,763] Trial 22 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 8, 'min_samples_split': 8, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.10445301234675547, 'min_impurity_decrease': 0.2549118647927189}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:02,160] Trial 23 finished with value: 0.5856156278909662 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 12, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.2340486968326449, 'min_impurity_decrease': 0.12161597897393306}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:02,421] Trial 24 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 12, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.4962229415516187, 'min_impurity_decrease': 0.14193156538106855}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:02,681] Trial 25 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.27862030005548816, 'min_impurity_decrease': 0.24041860380758118}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:02,940] Trial 26 finished with value: 0.4878982810180504 and parameters: {'criterion': 'gini', 'max_depth': 7, 'min_samples_split': 16, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.44079646928193805, 'min_impurity_decrease': 0.3359962576216289}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:03,210] Trial 27 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 12, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.25073773097551205, 'min_impurity_decrease': 0.09947502374086337}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:03,467] Trial 28 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 9, 'min_samples_split': 9, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.4075166561762072, 'min_impurity_decrease': 0.18887230396775156}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:03,724] Trial 29 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 11, 'min_samples_split': 13, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.3093611791388575, 'min_impurity_decrease': 0.6879328052924999}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:03,976] Trial 30 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 8, 'min_samples_split': 15, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.029038078655605926, 'min_impurity_decrease': 0.9708223413253263}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:04,401] Trial 31 finished with value: 0.610970073793593 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 4, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.20811588057741542, 'min_impurity_decrease': 0.08968505676257535}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:04,882] Trial 32 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 7, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.21658599113751817, 'min_impurity_decrease': 0.07013805364636741}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:05,555] Trial 33 finished with value: 0.6329360383812741 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 11, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.065549113489304, 'min_impurity_decrease': 0.1286432158301348}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:06,052] Trial 34 finished with value: 0.6063660420386825 and parameters: {'criterion': 'log_loss', 'max_depth': 7, 'min_samples_split': 20, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.06664727219621114, 'min_impurity_decrease': 0.22208237012325618}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:06,315] Trial 35 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 10, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.06330598540735383, 'min_impurity_decrease': 0.3153804654889647}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:06,587] Trial 36 finished with value: 0.4878982810180504 and parameters: {'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.019905404949795213, 'min_impurity_decrease': 0.12583097505564786}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:07,332] Trial 37 finished with value: 0.6195649835416622 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 3, 'min_samples_leaf': 10, 'min_weight_fraction_leaf': 0.11375906796336636, 'min_impurity_decrease': 0.05339111690156003}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:07,943] Trial 38 finished with value: 0.633291022138078 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 17, 'min_samples_leaf': 9, 'min_weight_fraction_leaf': 0.11017090618150657, 'min_impurity_decrease': 0.03655022391487544}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:08,785] Trial 39 finished with value: 0.5557969923194424 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 17, 'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0.05878339275324251, 'min_impurity_decrease': 0.04271032136510366}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:09,041] Trial 40 finished with value: 0.4878982810180504 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 14, 'min_samples_leaf': 9, 'min_weight_fraction_leaf': 0.08071731866605922, 'min_impurity_decrease': 0.1797939745132107}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:09,613] Trial 41 finished with value: 0.5046685742561476 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 19, 'min_samples_leaf': 10, 'min_weight_fraction_leaf': 0.11038661243851923, 'min_impurity_decrease': 0.05354329296371445}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:10,172] Trial 42 finished with value: 0.494524644478389 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 14, 'min_samples_leaf': 9, 'min_weight_fraction_leaf': 0.16564099010858493, 'min_impurity_decrease': 0.004813826738099686}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:10,788] Trial 43 finished with value: 0.6609259697510811 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 7, 'min_weight_fraction_leaf': 0.026020651937578458, 'min_impurity_decrease': 0.14076454662381493}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:11,047] Trial 44 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 18, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.02900640881187771, 'min_impurity_decrease': 0.4234435421162321}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:11,306] Trial 45 finished with value: 0.4878982810180504 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 7, 'min_weight_fraction_leaf': 0.0024360621696630524, 'min_impurity_decrease': 0.15389843949527546}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:11,564] Trial 46 finished with value: 0.4878982810180504 and parameters: {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 20, 'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0.04548797721489041, 'min_impurity_decrease': 0.30707244849198784}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:11,822] Trial 47 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 13, 'min_samples_split': 9, 'min_samples_leaf': 9, 'min_weight_fraction_leaf': 0.020144577085605773, 'min_impurity_decrease': 0.3901987232927469}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:12,086] Trial 48 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 10, 'min_samples_split': 16, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.07430331928659023, 'min_impurity_decrease': 0.20665703887368597}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:12,343] Trial 49 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 8, 'min_samples_split': 18, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.09705074832085409, 'min_impurity_decrease': 0.14432092776252045}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:12,842] Trial 50 finished with value: 0.4878982810180504 and parameters: {'criterion': 'gini', 'max_depth': 11, 'min_samples_split': 11, 'min_samples_leaf': 7, 'min_weight_fraction_leaf': 0.03792893611356134, 'min_impurity_decrease': 0.033261317475422686}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:13,321] Trial 51 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 3, 'min_samples_leaf': 10, 'min_weight_fraction_leaf': 0.12206307200992725, 'min_impurity_decrease': 0.05805121090615346}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:13,587] Trial 52 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 15, 'min_samples_leaf': 9, 'min_weight_fraction_leaf': 0.16419353462705225, 'min_impurity_decrease': 0.09714492971720039}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:14,309] Trial 53 finished with value: 0.5696198446677136 and parameters: {'criterion': 'log_loss', 'max_depth': 7, 'min_samples_split': 6, 'min_samples_leaf': 10, 'min_weight_fraction_leaf': 0.054324143362762076, 'min_impurity_decrease': 0.024763523241870648}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:16,831] Trial 54 finished with value: 0.5906499429874572 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 13, 'min_samples_leaf': 9, 'min_weight_fraction_leaf': 0.008159474344911374, 'min_impurity_decrease': 0.10667503499286127}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:17,460] Trial 55 finished with value: 0.6373679567995525 and parameters: {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 8, 'min_samples_leaf': 10, 'min_weight_fraction_leaf': 0.11030541298430627, 'min_impurity_decrease': 0.15661885532827077}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:17,727] Trial 56 finished with value: 0.4878982810180504 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 8, 'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0.08900113844851004, 'min_impurity_decrease': 0.15902787670194135}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:18,214] Trial 57 finished with value: 0.5148340182009854 and parameters: {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 7, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.04785082102510886, 'min_impurity_decrease': 0.2671184409672253}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:18,474] Trial 58 finished with value: 0.4878982810180504 and parameters: {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 10, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.15646009886020465, 'min_impurity_decrease': 0.2003168156933215}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:18,746] Trial 59 finished with value: 0.4878982810180504 and parameters: {'criterion': 'entropy', 'max_depth': 9, 'min_samples_split': 9, 'min_samples_leaf': 7, 'min_weight_fraction_leaf': 0.015836020292974, 'min_impurity_decrease': 0.28621985630156943}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:19,247] Trial 60 finished with value: 0.5512252318151503 and parameters: {'criterion': 'entropy', 'max_depth': 8, 'min_samples_split': 11, 'min_samples_leaf': 9, 'min_weight_fraction_leaf': 0.1354369132148509, 'min_impurity_decrease': 0.21410313492167446}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:19,808] Trial 61 finished with value: 0.5018287042017169 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 6, 'min_samples_leaf': 10, 'min_weight_fraction_leaf': 0.11586749230918444, 'min_impurity_decrease': 0.07055889414392671}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:20,456] Trial 62 finished with value: 0.5550547535552162 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 3, 'min_samples_leaf': 10, 'min_weight_fraction_leaf': 0.10169498576364913, 'min_impurity_decrease': 0.002513422363468659}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:20,717] Trial 63 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 17, 'min_samples_leaf': 10, 'min_weight_fraction_leaf': 0.18507001764843617, 'min_impurity_decrease': 0.11597540460800623}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:21,211] Trial 64 finished with value: 0.633291022138078 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 12, 'min_samples_leaf': 9, 'min_weight_fraction_leaf': 0.07356310478718567, 'min_impurity_decrease': 0.1639162070193102}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:21,883] Trial 65 finished with value: 0.5677373550482993 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 8, 'min_samples_leaf': 9, 'min_weight_fraction_leaf': 0.0338594818577731, 'min_impurity_decrease': 0.15912261146099488}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:22,370] Trial 66 finished with value: 0.4878982810180504 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 12, 'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0.07890259849968391, 'min_impurity_decrease': 0.1791897795987177}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:22,869] Trial 67 finished with value: 0.4970955874443321 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 10, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.06373409381334205, 'min_impurity_decrease': 0.2479680737083541}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:23,138] Trial 68 finished with value: 0.4878982810180504 and parameters: {'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 11, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.021802248213031936, 'min_impurity_decrease': 0.495319530156842}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:23,406] Trial 69 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 8, 'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0.08815103893319032, 'min_impurity_decrease': 0.1273609350320065}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:23,887] Trial 70 finished with value: 0.5202448312213592 and parameters: {'criterion': 'log_loss', 'max_depth': 9, 'min_samples_split': 13, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.04487089371560024, 'min_impurity_decrease': 0.08363677402531258}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:24,469] Trial 71 finished with value: 0.5148878036186829 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 12, 'min_samples_leaf': 10, 'min_weight_fraction_leaf': 0.12439784985574961, 'min_impurity_decrease': 0.03209697337354042}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:25,142] Trial 72 finished with value: 0.6318926012779416 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 16, 'min_samples_leaf': 9, 'min_weight_fraction_leaf': 0.07140060439357224, 'min_impurity_decrease': 0.09703847048789044}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:25,449] Trial 73 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 19, 'min_samples_leaf': 9, 'min_weight_fraction_leaf': 0.0754607577876698, 'min_impurity_decrease': 0.09987437644688818}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:25,995] Trial 74 finished with value: 0.5163077386458983 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 9, 'min_weight_fraction_leaf': 0.002299815366223864, 'min_impurity_decrease': 0.14448270207164945}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:26,278] Trial 75 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 16, 'min_samples_leaf': 9, 'min_weight_fraction_leaf': 0.056783791600140285, 'min_impurity_decrease': 0.23247951886410828}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:27,068] Trial 76 finished with value: 0.5485144467631936 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 9, 'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0.03165878229012109, 'min_impurity_decrease': 0.07343750142829505}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:27,355] Trial 77 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 18, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.09648764924668024, 'min_impurity_decrease': 0.7539817468875275}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:27,621] Trial 78 finished with value: 0.4878982810180504 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 15, 'min_samples_leaf': 7, 'min_weight_fraction_leaf': 0.07008470861744183, 'min_impurity_decrease': 0.17162725705619544}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:27,895] Trial 79 finished with value: 0.4878982810180504 and parameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 14, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.1508513446949793, 'min_impurity_decrease': 0.12605189277828602}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:28,785] Trial 80 finished with value: 0.6631742002108388 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 16, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.01701396989040868, 'min_impurity_decrease': 0.04832900710632898}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:29,551] Trial 81 finished with value: 0.6637981110561305 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 19, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.013311048820140996, 'min_impurity_decrease': 0.04089666550326358}. Best is trial 11 with value: 0.6655192444224521.
    [I 2026-06-24 19:23:30,663] Trial 82 finished with value: 0.6665626815257847 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 19, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.01424563059752934, 'min_impurity_decrease': 0.025830231499664725}. Best is trial 82 with value: 0.6665626815257847.
    [I 2026-06-24 19:23:32,070] Trial 83 finished with value: 0.6639379531421441 and parameters: {'criterion': 'log_loss', 'max_depth': 7, 'min_samples_split': 19, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.01058454440495124, 'min_impurity_decrease': 0.019639829487409888}. Best is trial 82 with value: 0.6665626815257847.
    [I 2026-06-24 19:23:33,227] Trial 84 finished with value: 0.6406596243626428 and parameters: {'criterion': 'log_loss', 'max_depth': 7, 'min_samples_split': 19, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.013590289757599459, 'min_impurity_decrease': 0.0235668941621522}. Best is trial 82 with value: 0.6665626815257847.
    [I 2026-06-24 19:23:34,573] Trial 85 finished with value: 0.6750500204384587 and parameters: {'criterion': 'log_loss', 'max_depth': 7, 'min_samples_split': 19, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.016553169831607848, 'min_impurity_decrease': 1.4983090750625427e-05}. Best is trial 85 with value: 0.6750500204384587.
    [I 2026-06-24 19:23:35,938] Trial 86 finished with value: 0.665852714012177 and parameters: {'criterion': 'log_loss', 'max_depth': 7, 'min_samples_split': 19, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.014295683779542427, 'min_impurity_decrease': 0.001828291334560285}. Best is trial 85 with value: 0.6750500204384587.
    [I 2026-06-24 19:23:37,229] Trial 87 finished with value: 0.6384006368193456 and parameters: {'criterion': 'log_loss', 'max_depth': 8, 'min_samples_split': 20, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.02409047788879247, 'min_impurity_decrease': 0.0016729837926834038}. Best is trial 85 with value: 0.6750500204384587.
    [I 2026-06-24 19:23:37,723] Trial 88 finished with value: 0.5202448312213592 and parameters: {'criterion': 'log_loss', 'max_depth': 7, 'min_samples_split': 18, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.0088566056691688, 'min_impurity_decrease': 0.05749031895383628}. Best is trial 85 with value: 0.6750500204384587.
    [I 2026-06-24 19:23:38,622] Trial 89 finished with value: 0.5541081302037392 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 19, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.0334764261087116, 'min_impurity_decrease': 0.020268905266811885}. Best is trial 85 with value: 0.6750500204384587.
    [I 2026-06-24 19:23:39,176] Trial 90 finished with value: 0.5285385426303221 and parameters: {'criterion': 'log_loss', 'max_depth': 8, 'min_samples_split': 20, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.0003046805697765609, 'min_impurity_decrease': 0.0490748310007973}. Best is trial 85 with value: 0.6750500204384587.
    [I 2026-06-24 19:23:40,506] Trial 91 finished with value: 0.6310427916783202 and parameters: {'criterion': 'log_loss', 'max_depth': 7, 'min_samples_split': 19, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.012129331759142763, 'min_impurity_decrease': 0.024488745251594043}. Best is trial 85 with value: 0.6750500204384587.
    [I 2026-06-24 19:23:40,970] Trial 92 finished with value: 0.5202448312213592 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 19, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.018662884524469854, 'min_impurity_decrease': 0.07335726213022896}. Best is trial 85 with value: 0.6750500204384587.
    [I 2026-06-24 19:23:41,683] Trial 93 finished with value: 0.6655837869236892 and parameters: {'criterion': 'log_loss', 'max_depth': 7, 'min_samples_split': 18, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.04593776955747916, 'min_impurity_decrease': 0.019125877068205815}. Best is trial 85 with value: 0.6750500204384587.
    [I 2026-06-24 19:23:42,785] Trial 94 finished with value: 0.6648415481594631 and parameters: {'criterion': 'log_loss', 'max_depth': 8, 'min_samples_split': 18, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.043639983461976076, 'min_impurity_decrease': 0.0009436028549898445}. Best is trial 85 with value: 0.6750500204384587.
    [I 2026-06-24 19:23:43,876] Trial 95 finished with value: 0.6529119425141455 and parameters: {'criterion': 'log_loss', 'max_depth': 8, 'min_samples_split': 18, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.04584389643988538, 'min_impurity_decrease': 0.003538057394150383}. Best is trial 85 with value: 0.6750500204384587.
    [I 2026-06-24 19:23:44,134] Trial 96 finished with value: 0.4878982810180504 and parameters: {'criterion': 'gini', 'max_depth': 9, 'min_samples_split': 18, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.03878320803428597, 'min_impurity_decrease': 0.04258651530107672}. Best is trial 85 with value: 0.6750500204384587.
    [I 2026-06-24 19:23:44,628] Trial 97 finished with value: 0.5202448312213592 and parameters: {'criterion': 'log_loss', 'max_depth': 7, 'min_samples_split': 20, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.025712504556578898, 'min_impurity_decrease': 0.07957280962553788}. Best is trial 85 with value: 0.6750500204384587.
    [I 2026-06-24 19:23:45,194] Trial 98 finished with value: 0.5835717820184592 and parameters: {'criterion': 'log_loss', 'max_depth': 8, 'min_samples_split': 19, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.05260910435790264, 'min_impurity_decrease': 0.043086313673848665}. Best is trial 85 with value: 0.6750500204384587.
    [I 2026-06-24 19:23:46,101] Trial 99 finished with value: 0.5966308814354252 and parameters: {'criterion': 'log_loss', 'max_depth': 7, 'min_samples_split': 17, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.039341635134416214, 'min_impurity_decrease': 0.016300132481570355}. Best is trial 85 with value: 0.6750500204384587.


Scikit-Learn already contains several classes for constructing ensemble models, but they expect to be fit to the data, not for the user to pass them models that have already been pre-fitted. We therefore roll our own. Fortunately, it's quite straightforward.


```python
def predict(X):
    """Predict using hard voting"""
    
    # Get predictions from each model
    predictions = np.fromiter((model.predict(X) for model in estimators), dtype=(np.int32, X.shape[0]))

    # Return the most common value in the array
    return scipy.stats.mode(predictions, axis=0)

def evaluate():
    mode_result = predict(X_test)
    y_pred = mode_result.mode

    # Use accuracy as the metric to evaluate the model
    return accuracy_score(y_true=y_test, y_pred=y_pred)
    
evaluate()
```




    0.48621808387046805



The performance of the ensemble is being dragged down by all the models that the Optuna library would have pruned. What if we adjusted the ensemble by discarding them?


```python
tmp = []
median = np.median(accuracies)

for model, accuracy in zip(estimators, accuracies):
    if accuracy > median:
        tmp.append(model)
        
estimators = tmp

evaluate()
```




    0.662986325654243



The accuracy of the ensemble is 66.29%. That's better than I had been previously able to achieve by including all of the decision trees. I may be able to improve the accuracy even further by increasing the number of trials for which to optimize using Optuna. However, I think it may be fruitful to consider a different class of models: Random Forests.

Most of the hyperparameters that we had passed to `DecisionTreeClassifier` are also accepted by `RandomForestClassifier`. That saves us from having to write new code. In fact, I could have combined this function with the objective function defined above, but I wouldn't want to disrupt the flow of this blog post.

The hyperparameters `n_estimators`, `bootstrap` and `max_samples` are new. These are set so that each tree is trained on 1/20 of the training data. This will save time.

Note that the default value of `bootstrap` is `True`.

As for `warm_start`, the documentation says, "When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest."

In other words, an instance of `RandomForestClassifier` can have new trees added onto it, a feature which we lean on in the following code snippet by inserting more trees into the ensemble at every iteration.

First, I start us off with 100 estimators and fit them. I don't want Optuna to be confused about the initial jump in performance and subsequent diminishing returns.


```python
n_estimators = 100
rf = RandomForestClassifier(
    n_estimators=n_estimators,
    warm_start=True,
    random_state=random_seed,
    max_samples=0.25,
    n_jobs=-1)
rf.fit(X_train, y_train)

last_accuracy = rf.score(X_val, y_val)

def objective(trial):

    global n_estimators
    global last_accuracy
    
    # Optuna estimates the best values for these hyperparameters
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0, 0.5)
    min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0, 1) 
    
    # Instantiate and fit the model
    n_estimators += 20
    rf.set_params(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        min_impurity_decrease=min_impurity_decrease,
    )
    rf.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = rf.score(X_val, y_val)
    improvement = accuracy - last_accuracy
    last_accuracy = accuracy

    return improvement

rf_study = optuna.create_study(direction='maximize')
rf_study.optimize(objective, n_trials=100)
```

    [I 2026-06-24 19:23:58,717] A new study created in memory with name: no-name-2c6a444f-dd09-4400-9120-2b54f4fc9e4a
    [I 2026-06-24 19:23:59,725] Trial 0 finished with value: -0.002463372130547925 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 18, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.37509085942116693, 'min_impurity_decrease': 0.0637868861614519}. Best is trial 0 with value: -0.002463372130547925.
    [I 2026-06-24 19:24:00,828] Trial 1 finished with value: -0.003496052150340989 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 17, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.27637547421468833, 'min_impurity_decrease': 0.28025951728709275}. Best is trial 0 with value: -0.002463372130547925.
    [I 2026-06-24 19:24:01,870] Trial 2 finished with value: -0.0037649792388287917 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 19, 'min_samples_leaf': 7, 'min_weight_fraction_leaf': 0.49640026314909036, 'min_impurity_decrease': 0.46093861588037566}. Best is trial 0 with value: -0.002463372130547925.
    [I 2026-06-24 19:24:02,889] Trial 3 finished with value: -0.004582517587831569 and parameters: {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 7, 'min_samples_leaf': 10, 'min_weight_fraction_leaf': 0.39080014898173543, 'min_impurity_decrease': 0.25480322831533697}. Best is trial 0 with value: -0.002463372130547925.
    [I 2026-06-24 19:24:03,954] Trial 4 finished with value: -0.005894881779651917 and parameters: {'criterion': 'log_loss', 'max_depth': 11, 'min_samples_split': 13, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.38677710616006095, 'min_impurity_decrease': 0.7476516574320726}. Best is trial 0 with value: -0.002463372130547925.
    [I 2026-06-24 19:24:05,038] Trial 5 finished with value: -0.008627180998687645 and parameters: {'criterion': 'log_loss', 'max_depth': 12, 'min_samples_split': 16, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.10092181389329213, 'min_impurity_decrease': 0.29017641638605884}. Best is trial 0 with value: -0.002463372130547925.
    [I 2026-06-24 19:24:06,122] Trial 6 finished with value: -0.01242443148813488 and parameters: {'criterion': 'entropy', 'max_depth': 9, 'min_samples_split': 15, 'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0.027461653051046198, 'min_impurity_decrease': 0.49006696425961105}. Best is trial 0 with value: -0.002463372130547925.
    [I 2026-06-24 19:24:07,229] Trial 7 finished with value: -0.015070674038854648 and parameters: {'criterion': 'log_loss', 'max_depth': 11, 'min_samples_split': 10, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.38150232574335874, 'min_impurity_decrease': 0.2725703371530661}. Best is trial 0 with value: -0.002463372130547925.
    [I 2026-06-24 19:24:08,352] Trial 8 finished with value: -0.01726511908091477 and parameters: {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 5, 'min_samples_leaf': 10, 'min_weight_fraction_leaf': 0.1520694051108235, 'min_impurity_decrease': 0.8723436890358002}. Best is trial 0 with value: -0.002463372130547925.
    [I 2026-06-24 19:24:09,482] Trial 9 finished with value: -0.02277274585314426 and parameters: {'criterion': 'log_loss', 'max_depth': 7, 'min_samples_split': 12, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.3680405090372734, 'min_impurity_decrease': 0.3991429974888686}. Best is trial 0 with value: -0.002463372130547925.
    [I 2026-06-24 19:24:10,795] Trial 10 finished with value: -0.025053247563520564 and parameters: {'criterion': 'gini', 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.21787040324078152, 'min_impurity_decrease': 0.00024706506994531807}. Best is trial 0 with value: -0.002463372130547925.
    [I 2026-06-24 19:24:12,025] Trial 11 finished with value: -0.02150340999548206 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 19, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.26663050345883044, 'min_impurity_decrease': 0.026032473240972}. Best is trial 0 with value: -0.002463372130547925.
    [I 2026-06-24 19:24:13,276] Trial 12 finished with value: -0.003022740474602492 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 20, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.2828431123169739, 'min_impurity_decrease': 0.11856340925253843}. Best is trial 0 with value: -0.002463372130547925.
    [I 2026-06-24 19:24:14,510] Trial 13 finished with value: -0.0026569996342592495 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 20, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.4951784035427042, 'min_impurity_decrease': 0.12159566863768556}. Best is trial 0 with value: -0.002463372130547925.
    [I 2026-06-24 19:24:15,816] Trial 14 finished with value: -0.002355801295152782 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.4742619582254747, 'min_impurity_decrease': 0.1402888259157592}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:17,108] Trial 15 finished with value: -0.002527914631785033 and parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 14, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.4384172356757767, 'min_impurity_decrease': 0.12734015944913868}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:18,447] Trial 16 finished with value: -0.0031195542264581544 and parameters: {'criterion': 'entropy', 'max_depth': 9, 'min_samples_split': 17, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.3232324164858168, 'min_impurity_decrease': 0.6784333281899733}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:19,780] Trial 17 finished with value: -0.002990469223983938 and parameters: {'criterion': 'gini', 'max_depth': 9, 'min_samples_split': 11, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.4448366368838968, 'min_impurity_decrease': 0.16706463165160268}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:21,140] Trial 18 finished with value: -0.0036466513198941675 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.4506833243559913, 'min_impurity_decrease': 0.5646553958805762}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:24,351] Trial 19 finished with value: -0.002355801295152893 and parameters: {'criterion': 'entropy', 'max_depth': 8, 'min_samples_split': 9, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.3355157866993127, 'min_impurity_decrease': 0.002909341681439359}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:25,868] Trial 20 finished with value: -0.0032701533960112217 and parameters: {'criterion': 'entropy', 'max_depth': 8, 'min_samples_split': 9, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.20081932279449574, 'min_impurity_decrease': 0.19841407008965348}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:27,321] Trial 21 finished with value: -0.002355801295152893 and parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 2, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.3258318755655925, 'min_impurity_decrease': 0.047128038424321096}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:28,808] Trial 22 finished with value: -0.003033497558142084 and parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 3, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.32555264934179434, 'min_impurity_decrease': 0.01827490909005175}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:30,287] Trial 23 finished with value: -0.003829521740065789 and parameters: {'criterion': 'entropy', 'max_depth': 8, 'min_samples_split': 8, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.33306134795759584, 'min_impurity_decrease': 0.3660098661925487}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:31,828] Trial 24 finished with value: -0.00433510466642284 and parameters: {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 5, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.23112883620831481, 'min_impurity_decrease': 0.20215428200233104}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:33,348] Trial 25 finished with value: -0.0046470600890686775 and parameters: {'criterion': 'gini', 'max_depth': 6, 'min_samples_split': 6, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.41660964773389236, 'min_impurity_decrease': 0.08992402300992063}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:34,947] Trial 26 finished with value: -0.0035713517351175783 and parameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.3102713099624924, 'min_impurity_decrease': 0.19681661457893557}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:36,547] Trial 27 finished with value: -0.004690088423226713 and parameters: {'criterion': 'entropy', 'max_depth': 8, 'min_samples_split': 2, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.18318325359058804, 'min_impurity_decrease': 0.989199531344853}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:38,159] Trial 28 finished with value: -0.0051096146812675824 and parameters: {'criterion': 'entropy', 'max_depth': 15, 'min_samples_split': 8, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.3444586870699229, 'min_impurity_decrease': 0.35533008523740467}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:39,827] Trial 29 finished with value: -0.005303242184978796 and parameters: {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 11, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.24113996273891358, 'min_impurity_decrease': 0.068540904560518}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:41,482] Trial 30 finished with value: -0.003463780899722435 and parameters: {'criterion': 'gini', 'max_depth': 13, 'min_samples_split': 14, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.13785198041794128, 'min_impurity_decrease': 0.07684517207013257}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:43,267] Trial 31 finished with value: -0.0041199629956326644 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 18, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.2790104875025435, 'min_impurity_decrease': 0.002338576680177895}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:44,939] Trial 32 finished with value: -0.00500204384587255 and parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 18, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.47926376285938993, 'min_impurity_decrease': 0.14953527510273684}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:46,675] Trial 33 finished with value: -0.00492674426109585 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 15, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.4111562310603123, 'min_impurity_decrease': 0.06117325799143175}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:48,518] Trial 34 finished with value: -0.00533551343559735 and parameters: {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 7, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.35776128425833786, 'min_impurity_decrease': 0.2451304321135578}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:50,314] Trial 35 finished with value: -0.004808416342161337 and parameters: {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 16, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.4091683037240057, 'min_impurity_decrease': 0.07783663174253148}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:52,309] Trial 36 finished with value: -0.003947849659000413 and parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 13, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.2951854277931003, 'min_impurity_decrease': 0.23839140189561173}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:54,231] Trial 37 finished with value: -0.007314816806867408 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 18, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.46750798278158345, 'min_impurity_decrease': 0.29267285831695117}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:56,146] Trial 38 finished with value: -0.008734751834082677 and parameters: {'criterion': 'gini', 'max_depth': 7, 'min_samples_split': 16, 'min_samples_leaf': 9, 'min_weight_fraction_leaf': 0.378962911915205, 'min_impurity_decrease': 0.31600673231647386}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:24:59,900] Trial 39 finished with value: -0.007196488887932673 and parameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.24961568462150818, 'min_impurity_decrease': 0.1615953589737063}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:25:01,875] Trial 40 finished with value: -0.007788128482605794 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 19, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.006361866830458807, 'min_impurity_decrease': 0.054620514020711476}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:25:03,840] Trial 41 finished with value: -0.006873776381747465 and parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 14, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.4331234416815187, 'min_impurity_decrease': 0.12484837570745991}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:25:05,888] Trial 42 finished with value: -0.010305286030851235 and parameters: {'criterion': 'entropy', 'max_depth': 8, 'min_samples_split': 15, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.39436465853036573, 'min_impurity_decrease': 0.12734129146256218}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:25:08,014] Trial 43 finished with value: -0.009520018932467122 and parameters: {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 13, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.36105647602240837, 'min_impurity_decrease': 0.03695102160632004}. Best is trial 14 with value: -0.002355801295152782.
    [I 2026-06-24 19:25:10,093] Trial 44 finished with value: -0.0011832791893461314 and parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 16, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.49842627337031453, 'min_impurity_decrease': 0.21431109236967727}. Best is trial 44 with value: -0.0011832791893461314.
    [I 2026-06-24 19:25:12,202] Trial 45 finished with value: -0.01009014436006106 and parameters: {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 17, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.4921387127745064, 'min_impurity_decrease': 0.23182893962502338}. Best is trial 44 with value: -0.0011832791893461314.
    [I 2026-06-24 19:25:14,296] Trial 46 finished with value: -0.015630042382909215 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 20, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.4711903203936137, 'min_impurity_decrease': 0.1737514194599606}. Best is trial 44 with value: -0.0011832791893461314.
    [I 2026-06-24 19:25:16,375] Trial 47 finished with value: -0.01692089240765038 and parameters: {'criterion': 'log_loss', 'max_depth': 7, 'min_samples_split': 19, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.30723685519875077, 'min_impurity_decrease': 0.48470295138478736}. Best is trial 44 with value: -0.0011832791893461314.
    [I 2026-06-24 19:25:18,479] Trial 48 finished with value: -0.012015662313633602 and parameters: {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 12, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.45432066172678043, 'min_impurity_decrease': 0.10174379864137914}. Best is trial 44 with value: -0.0011832791893461314.
    [I 2026-06-24 19:25:20,648] Trial 49 finished with value: -0.0046685742561476395 and parameters: {'criterion': 'gini', 'max_depth': 6, 'min_samples_split': 16, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.06913126604851813, 'min_impurity_decrease': 0.03616965368116118}. Best is trial 44 with value: -0.0011832791893461314.
    [I 2026-06-24 19:25:22,952] Trial 50 finished with value: -0.021965964587680964 and parameters: {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 4, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.4248949266652946, 'min_impurity_decrease': 0.4164093477693361}. Best is trial 44 with value: -0.0011832791893461314.
    [I 2026-06-24 19:25:25,135] Trial 51 finished with value: -0.017146791161980146 and parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 15, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.3884949259443027, 'min_impurity_decrease': 0.004456380751704916}. Best is trial 44 with value: -0.0011832791893461314.
    [I 2026-06-24 19:25:27,401] Trial 52 finished with value: -0.01917987995094772 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 18, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.4443334073762491, 'min_impurity_decrease': 0.15015932012033198}. Best is trial 44 with value: -0.0011832791893461314.
    [I 2026-06-24 19:25:29,723] Trial 53 finished with value: -0.008110840988791113 and parameters: {'criterion': 'entropy', 'max_depth': 8, 'min_samples_split': 14, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.4892895409365487, 'min_impurity_decrease': 0.10736176488861954}. Best is trial 44 with value: -0.0011832791893461314.
    [I 2026-06-24 19:25:33,821] Trial 54 finished with value: -0.010843140207826896 and parameters: {'criterion': 'entropy', 'max_depth': 9, 'min_samples_split': 16, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.34623072678160227, 'min_impurity_decrease': 0.20265666277651345}. Best is trial 44 with value: -0.0011832791893461314.
    [I 2026-06-24 19:25:36,134] Trial 55 finished with value: -0.005324756352057813 and parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 9, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.4630608546809942, 'min_impurity_decrease': 0.290759402432427}. Best is trial 44 with value: -0.0011832791893461314.
    [I 2026-06-24 19:25:38,492] Trial 56 finished with value: -0.0005808825111335847 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.4005697245613851, 'min_impurity_decrease': 0.055057655401156425}. Best is trial 56 with value: -0.0005808825111335847.
    [I 2026-06-24 19:25:40,785] Trial 57 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 19, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.40079058997628836, 'min_impurity_decrease': 0.044113658742854334}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:25:43,351] Trial 58 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.4021460349958285, 'min_impurity_decrease': 0.04781134274391041}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:25:45,750] Trial 59 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.39732169819464985, 'min_impurity_decrease': 0.036187685270910826}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:25:48,259] Trial 60 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.39933384985792475, 'min_impurity_decrease': 0.09392990004668274}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:25:50,730] Trial 61 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.4050645068350336, 'min_impurity_decrease': 0.0908596495029869}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:25:53,155] Trial 62 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 17, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.39783668539275163, 'min_impurity_decrease': 0.08340007503669405}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:25:55,617] Trial 63 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 19, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.4025686227765634, 'min_impurity_decrease': 0.09317559793844413}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:25:58,107] Trial 64 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 19, 'min_samples_leaf': 7, 'min_weight_fraction_leaf': 0.3745321166981747, 'min_impurity_decrease': 0.09610327040184742}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:00,695] Trial 65 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 20, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.4284266293150925, 'min_impurity_decrease': 0.04184175195960138}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:03,305] Trial 66 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 17, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.3985732272510746, 'min_impurity_decrease': 0.16457012404970617}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:07,811] Trial 67 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 18, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.36912208436931054, 'min_impurity_decrease': 0.09399627757996784}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:10,582] Trial 68 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 19, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.419374232633046, 'min_impurity_decrease': 0.028591127784406527}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:13,257] Trial 69 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 19, 'min_samples_leaf': 7, 'min_weight_fraction_leaf': 0.383336400086092, 'min_impurity_decrease': 0.7810478015323197}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:16,030] Trial 70 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 18, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.3536727827524967, 'min_impurity_decrease': 0.07216508183630202}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:18,688] Trial 71 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 19, 'min_samples_leaf': 7, 'min_weight_fraction_leaf': 0.3745581664113634, 'min_impurity_decrease': 0.09524126777589192}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:21,330] Trial 72 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 17, 'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0.40595142363166153, 'min_impurity_decrease': 0.12776014287653045}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:24,127] Trial 73 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 20, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.43699210692811336, 'min_impurity_decrease': 0.18714029473735583}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:26,970] Trial 74 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 17, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.3702095018997533, 'min_impurity_decrease': 0.10726720193850643}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:29,909] Trial 75 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 18, 'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0.3855647324076468, 'min_impurity_decrease': 0.019893069139176253}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:32,872] Trial 76 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 19, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.4186256688227415, 'min_impurity_decrease': 0.07598035015119457}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:35,800] Trial 77 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 20, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.33863393643215567, 'min_impurity_decrease': 0.1488950757694963}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:38,819] Trial 78 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.39834121909997006, 'min_impurity_decrease': 0.5932689121337046}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:43,617] Trial 79 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 16, 'min_samples_leaf': 7, 'min_weight_fraction_leaf': 0.3603884362819875, 'min_impurity_decrease': 0.06661407049189055}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:46,593] Trial 80 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 14, 'min_samples_split': 18, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.4480317035178996, 'min_impurity_decrease': 0.0026945205042975817}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:49,718] Trial 81 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 20, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.43008019424299143, 'min_impurity_decrease': 0.04519756013767649}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:52,764] Trial 82 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 19, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.41192348184653105, 'min_impurity_decrease': 0.037467626467139685}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:55,726] Trial 83 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 20, 'min_samples_leaf': 7, 'min_weight_fraction_leaf': 0.4300296518468413, 'min_impurity_decrease': 0.13003813167493367}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:26:58,662] Trial 84 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 18, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.380665957849, 'min_impurity_decrease': 0.09580437964441355}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:27:01,598] Trial 85 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 19, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.3951791853149382, 'min_impurity_decrease': 0.049091172964409326}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:27:04,655] Trial 86 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 16, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.4554709228217944, 'min_impurity_decrease': 0.1787243359400109}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:27:07,745] Trial 87 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 19, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.4068158988091382, 'min_impurity_decrease': 0.08171803656755523}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:27:11,137] Trial 88 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 15, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.4253896457709222, 'min_impurity_decrease': 0.025724039079096445}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:27:16,411] Trial 89 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 20, 'min_samples_leaf': 9, 'min_weight_fraction_leaf': 0.34679408543592033, 'min_impurity_decrease': 0.11936963141238471}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:27:19,549] Trial 90 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 18, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.32310090037626643, 'min_impurity_decrease': 0.1446792385563959}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:27:23,184] Trial 91 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 17, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.39555487419994906, 'min_impurity_decrease': 0.17111277576120526}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:27:26,348] Trial 92 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 17, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.3680187988694129, 'min_impurity_decrease': 0.05696905537808408}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:27:29,749] Trial 93 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 18, 'min_samples_leaf': 7, 'min_weight_fraction_leaf': 0.4193952262804675, 'min_impurity_decrease': 0.26524187663701276}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:27:32,924] Trial 94 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 16, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.38854295298158414, 'min_impurity_decrease': 0.2156871122899327}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:27:36,174] Trial 95 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 17, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.4401966786288225, 'min_impurity_decrease': 0.10768694599844718}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:27:39,473] Trial 96 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 7, 'min_weight_fraction_leaf': 0.40502740573547996, 'min_impurity_decrease': 0.08169811768134458}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:27:42,967] Trial 97 finished with value: 0.0 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 15, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.3788113587181597, 'min_impurity_decrease': 0.022716213071311716}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:27:46,608] Trial 98 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 16, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0.35757812876772627, 'min_impurity_decrease': 0.1539728314389629}. Best is trial 57 with value: 0.0.
    [I 2026-06-24 19:27:51,844] Trial 99 finished with value: 0.0 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 18, 'min_samples_leaf': 6, 'min_weight_fraction_leaf': 0.4001635798783094, 'min_impurity_decrease': 0.06318615220633152}. Best is trial 57 with value: 0.0.


Given the sort of numbers I'm seeing in the above, I'm not holding my breath for the ensemble to have been improved.


```python
rf.score(X_test, y_test)
```




    0.48621808387046805



Looks like I was right not to get my hopes up. How many individual trees are inside the forest, anyway?


```python
rf.n_estimators
```




    2100



The performance is dismal for how many trees we fit. Nevertheless, we must soldier on! Should we again throw out the models not pulling their weight to see if that bumps up the performance of the whole?

We could, but I have a different strategy in mind.

This is a popular dataset with multiple implementations available on the internet. I've therefore done cursory research on the kind of performance that I can expect a Random Forest to have on the Forest Cover Type dataset. I'm falling far short of the ideal. What if I just used all the default values of `RandomForestClassifier` except `n_estimators`?


```python
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
```




    0.951274924055317



Just as I expected. I was so heady with the power of this advanced hyperparameter tuning library that I neglected the basics, which were all that were ever required.

A certain quote comes to mind.

> I have not failed. I've just found 10,000 ways that won't work.

― Thomas A. Edison

Still, I want to get a set of improved hyperparameters out of this, at least, so I'll find if there is an ideal maximum sample number.



```python
n_estimators = 100
rf = RandomForestClassifier(
    n_estimators=n_estimators,
    warm_start=True,
    random_state=random_seed,
    max_samples=0.25,
    n_jobs=-1)
rf.fit(X_train, y_train)

last_accuracy = rf.score(X_val, y_val)

def objective(trial):
    global n_estimators
    global last_accuracy
    
    # Optuna estimates the best values for these hyperparameters
    denominator = trial.suggest_int('denominator', 2, 20)
    
    # Instantiate and fit the model
    n_estimators += 20
    rf.set_params(
        n_estimators=n_estimators,
        max_samples=1.0/denominator,
    )
    rf.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = rf.score(X_val, y_val)
    print(f"Denominator: {denominator}, accuracy: {accuracy}")
    improvement = accuracy - last_accuracy
    last_accuracy = accuracy

    return improvement

rf_study = optuna.create_study(direction='maximize')
rf_study.optimize(objective, n_trials=10)
```

    [I 2026-06-24 19:31:02,305] A new study created in memory with name: no-name-fbd7fd91-d3e2-48f0-acac-7e42d80a7464
    [I 2026-06-24 19:31:03,702] Trial 0 finished with value: -0.004087691745013999 and parameters: {'denominator': 14}. Best is trial 0 with value: -0.004087691745013999.


    Denominator: 14, accuracy: 0.9225489985155225


    [I 2026-06-24 19:31:05,666] Trial 1 finished with value: -0.0030550117252211573 and parameters: {'denominator': 11}. Best is trial 1 with value: -0.0030550117252211573.


    Denominator: 11, accuracy: 0.9194939867903014


    [I 2026-06-24 19:31:08,721] Trial 2 finished with value: 0.0017749187840193636 and parameters: {'denominator': 4}. Best is trial 2 with value: 0.0017749187840193636.


    Denominator: 4, accuracy: 0.9212689055743207


    [I 2026-06-24 19:31:10,221] Trial 3 finished with value: -0.002861384221509944 and parameters: {'denominator': 16}. Best is trial 2 with value: 0.0017749187840193636.


    Denominator: 16, accuracy: 0.9184075213528108


    [I 2026-06-24 19:31:14,991] Trial 4 finished with value: 0.0012585787741228316 and parameters: {'denominator': 4}. Best is trial 2 with value: 0.0017749187840193636.


    Denominator: 4, accuracy: 0.9196661001269336


    [I 2026-06-24 19:31:17,008] Trial 5 finished with value: -0.001452206277833934 and parameters: {'denominator': 9}. Best is trial 2 with value: 0.0017749187840193636.


    Denominator: 9, accuracy: 0.9182138938490997


    [I 2026-06-24 19:31:18,814] Trial 6 finished with value: -0.0017749187840193636 and parameters: {'denominator': 14}. Best is trial 2 with value: 0.0017749187840193636.


    Denominator: 14, accuracy: 0.9164389750650803


    [I 2026-06-24 19:31:21,608] Trial 7 finished with value: 0.00036574084034335375 and parameters: {'denominator': 5}. Best is trial 2 with value: 0.0017749187840193636.


    Denominator: 5, accuracy: 0.9168047159054237


    [I 2026-06-24 19:31:23,705] Trial 8 finished with value: -0.00150599169553145 and parameters: {'denominator': 13}. Best is trial 2 with value: 0.0017749187840193636.


    Denominator: 13, accuracy: 0.9152987242098922


    [I 2026-06-24 19:31:26,577] Trial 9 finished with value: -2.1514167079073054e-05 and parameters: {'denominator': 7}. Best is trial 2 with value: 0.0017749187840193636.


    Denominator: 7, accuracy: 0.9152772100428131


Despite the fact that the number of estimators are equal, this did not perform as well as the classic Random Forest Classifier with no value set for `max_samples`.

Still, maybe what the classic is telling us is that our model is underfitting.

It's fortunate, therefore, that Optuna allows us to perform subsequent optimization runs that build off of the previous.


```python
rf_study.optimize(objective, n_trials=10)
```

    [I 2026-06-24 19:31:28,719] Trial 10 finished with value: -0.0017534046169401796 and parameters: {'denominator': 19}. Best is trial 2 with value: 0.0017749187840193636.


    Denominator: 19, accuracy: 0.913523805425873


    [I 2026-06-24 19:31:35,133] Trial 11 finished with value: 0.0027000279684171735 and parameters: {'denominator': 2}. Best is trial 11 with value: 0.0027000279684171735.


    Denominator: 2, accuracy: 0.9162238333942901


    [I 2026-06-24 19:31:41,248] Trial 12 finished with value: 0.002065360039586017 and parameters: {'denominator': 2}. Best is trial 11 with value: 0.0027000279684171735.


    Denominator: 2, accuracy: 0.9182891934338762


    [I 2026-06-24 19:31:48,178] Trial 13 finished with value: 0.00191476087003295 and parameters: {'denominator': 2}. Best is trial 11 with value: 0.0027000279684171735.


    Denominator: 2, accuracy: 0.9202039543039091


    [I 2026-06-24 19:31:53,339] Trial 14 finished with value: 0.001936275037111912 and parameters: {'denominator': 2}. Best is trial 11 with value: 0.0027000279684171735.


    Denominator: 2, accuracy: 0.922140229341021


    [I 2026-06-24 19:31:56,493] Trial 15 finished with value: -0.0003872550074223158 and parameters: {'denominator': 7}. Best is trial 11 with value: 0.0027000279684171735.


    Denominator: 7, accuracy: 0.9217529743335987


    [I 2026-06-24 19:32:01,533] Trial 16 finished with value: 0.0013016071082807557 and parameters: {'denominator': 2}. Best is trial 11 with value: 0.0027000279684171735.


    Denominator: 2, accuracy: 0.9230545814418795


    [I 2026-06-24 19:32:04,874] Trial 17 finished with value: 0.00011832791893462424 and parameters: {'denominator': 6}. Best is trial 11 with value: 0.0027000279684171735.


    Denominator: 6, accuracy: 0.9231729093608141


    [I 2026-06-24 19:32:09,170] Trial 18 finished with value: -0.00015059916955317831 and parameters: {'denominator': 4}. Best is trial 11 with value: 0.0027000279684171735.


    Denominator: 4, accuracy: 0.9230223101912609


    [I 2026-06-24 19:32:12,844] Trial 19 finished with value: -0.000484068759277978 and parameters: {'denominator': 9}. Best is trial 11 with value: 0.0027000279684171735.


    Denominator: 9, accuracy: 0.9225382414319829


It's interesting to note that the improvement being negative is correlated with the magnitude of the denominator.

Let's calculate the accuracy on the test set.


```python
rf.score(X_test, y_test)
```




    0.9227386556285122



We still couldn't beat the classic . . .


```python
rf.n_estimators
```




    500



. . . despite the number of estimators being higher.

In this post, we explored the use of Optuna for constructing an ensemble and found that we couldn't beat the default settings of Scikit-Learn's `RandomForestClassifier`. I feel confident that in a future post I'll find a better use of Optuna.

## Bibliography

Akiba, Takuya, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. “Optuna: A Next-Generation Hyperparameter Optimization Framework.” Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2019.

Blackard, J. (1998). Covertype [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50K5N.

Pedregosa, F., G. Varoquaux, A. Gramfort, et al. “Scikit-Learn: Machine Learning in Python.” Journal of Machine Learning Research 12 (2011): 2825–30.
