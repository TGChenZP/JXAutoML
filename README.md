# JXAutoML
```
Author GitHub: https://github.com/TGChenZP
```

*Please cite when using this package for research and other machine learning purposes*

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Classes](#classes)
    - [NingXiang](#ningxiang)
    - [JiaoCheng](#jiaocheng)
    - [JiaoChengB](#jiaochengb)
    - [YangZhouB](#yangzhoub)
4. [Usage Examples](#usage-examples)
    - [NingXiang](#ningxiang-example)
    - [JiaoCheng](#jiaocheng-example)
    - [JiaoChengB](#jiaochengb-example)
    - [YangZhouB](#yangzhou-example)

# Introduction
A package that provides smart tuning using greedy algorithms, and also enables feature selection as part of hyperparameter tuning.

# Installation
```bash
pip install JXAutoML 
```

# Classes
## NingXiang
>Feature importance extractor using Linear Regression and Random Forest

**Background**

The purpose of this package is to provide an ordering of discrete feature combinations for tuning packages JiaoCheng and YangZhou so that feature combinations can participate in tuning like a hyperparameter in a meaningful way.

The package performs feature combination ordering by linear regression √(r2) and random forest regressor feature importance, which represents the multi-feature vs label equivalent of pearson’s coefficient and NMI respectively.

## Algorithm Description

Linear Regression sqrt(r²):

- Starting with no features in the combination, iteratively add the feature that will increase r² of the resulting y~X linear model using training features by the most.

  Then, `{combo: NingXiang score}` will be `{X, sqrt(r² of y~X)}`

Random Forest feature importance:

- First use 
```python 
RandomForest(n_estimators = 100, max_depth = 12, max_features = 0.75, random_state = self._seed, ccp_alpha = 0, max_samples = 0.75)
``` 
to build the model based on training data.

- Then, using this model’s feature importance, iteratively add features in reverse importance, and set NingXiang score as the sum of this set of features’ importance.

### Note:
- Under both use cases, the NingXiang score will be between [0, 1], however, the linear regression use case is not guaranteed to reach 1.
- Classification models should only use Random Forest feature importance.


### Methods
| **Method**                                                   | **Description**                                                                                                                                                                  | **Parameters**                                                                                                                                                                                 |
|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `min_features = 0, gap = 1`                                  | Builds a linear regression model and gets NingXiang score based on the square root of R², iteratively adding the feature that increases R² the most each time. <br> Can set the number of features in the minimum feature combo (to avoid having a combo of just 1 or 2 features, as some models can’t train with too few features). <br> Can set the gap between the number of features in neighboring combinations (useful for Natural Language Processing where there are too many features to try all). | `min_features` (`int`): The minimum number of features in a combination. <br> `gap` (`int`): The gap between the number of features in neighboring combinations.                                                                 |
| `get_rf_based_feature_combinations(min_features = 0, gap = 1, n_jobs = 1)` | Builds a random forest model and gets NingXiang output based on feature importance. <br> Can set the number of features in the minimum feature combo. <br> Can set the gap between the number of features in neighboring combinations. | `min_features` (`int`): The minimum number of features in a combination. <br> `gap` (`int`): The gap between the number of features in neighboring combinations. <br> `n_jobs` (`int`): The number of jobs to run in parallel. |
| `get_rf_based_feature_combinations_from_feature_importance(feature_importance = None, min_features = 0, gap = 1)` | Uses ready-made feature importance to create NingXiang output. <br> Can set the number of features in the minimum feature combo. <br> Can set the gap between the number of features in neighboring combinations. | `feature_importance` (`dict` of `str:float`): Pre-calculated feature importance. <br> `min_features` (`int`): The minimum number of features in a combination. <br> `gap` (`int`): The gap between the number of features in neighboring combinations. |
| `show_rf_stats()`                                            | Displays the random forest feature importance dataframe and the validation score (if the validation score was inputted).                                                        | N/A                                                                                                                                                                                             |
| `export_ningxiang_output(address)`                           | Exports the current NingXiang output object as a pickle object.                                                                                                                  | `address` (`str`): The address where the pickle object will be saved. Does not need to include `.pickle`.                                                                                       |

### Attributes
| **Attribute**              | **Type**            |
|----------------------------|---------------------|
| `train_x`                  | `DataFrame`         |
| `train_y`                  | `Series`            |
| `clf_type`                 | `str`  - 'Regression' or 'Classification'       |
| `ningxiang_output`         | `dict`              |
| `object_saving_address`    | `str`               |


## JiaoCheng
>Ultra greedy hyperparameter tuning algorithm

**Background**

The purpose of this package is to provide a framework for feature-by-feature tuning, a different (and in most cases faster but less accurate) method compared to JiXi, YangZhou etc.

Sometimes, a data scientist would be stuck in the midst of data cleaning, but would like to get a glimpse of how well this data is currently performing as a benchmark, and hence does not need to necessarily find the global maximum in the field space. JiaoCheng, being more greedy and hence training less combinations and taking less time than JiXi, is suitable for this purpose.

The package takes in X and y data for train, validate and test as DataFrame, as well as a dictionary of {hyperparameters name -> string: hyperparameter values as a list}, dictionary of default values for each hyperparameter and list of order of features, and autogenerates all combinations of these hyperparameters to be tuned.

JiaoCheng starts at the default values combination, and searches through different values of first hyperparameter whilst holding other hyperparameter values constant. The maximum combination from this search gets updated as the new ‘default value combination’ (now called ‘current max combination’) and the second hyperparameter is searched through holding other hyperparameter values of the ‘current max combination’ fixed. Once all hyperparameter have been searched through in this manner, if the ‘current max combination’ is the same as that before this round of all hyperparameter being searched, then the algorithm is terminated. Else, another round of search is undertaken.

The idea was taken from the Gibbs Sampling Algorithm in statistics.

### Methods
| **Method**                                        | **Description**                                                                                                                                                                  | **Parameters**                                                                                                                                                                                 |
|--------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `JiaoCheng()`                                    | Initialisation of the JiaoCheng object.                                                                                                                                           | N/A                                                                                                                                                                                             |
| `read_in_data(train_x, train_y, val_x, val_y, test_x, test_y)` | Reads in Train Test Split data.                                                                                                                                                  | `train_x` (`pd.DataFrame`): Training data features. <br> `train_y` (`pd.Series`): Training data labels. <br> `val_x` (`pd.DataFrame`): Validation data features. <br> `val_y` (`pd.Series`): Validation data labels. <br> `test_x` (`pd.DataFrame`): Test data features. <br> `test_y` (`pd.Series`): Test data labels. |
| `read_in_model(model, type, optimised_metric, pytorch_model)` | Reads in the underlying model class to tune for optimal parameters; also reads in what metric to optimise for, and whether the model is a PyTorch class model.                    | `model` (Any model class): Must allow `.fit()` and `.predict()`. <br> `type` (`str`): Either "Classification" or "Regression". <br> `optimised_metric` (`str`): If `type = 'Regression'`, must be in `['r2', 'rmse', 'mape']`; if `type = 'Classification'`, must be in `['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy', 'AP', 'AUC']`. <br> `pytorch_model` (`bool`): Whether the model is a PyTorch class model. |
| `set_hyperparameters(parameter_choices)`         | Reads in the different values of each hyperparameter to try. Function will automatically generate each combination.                                                               | `parameter_choices` (`dict` of `str:list`): Hyperparameter names (as defined in model class) and their sorted values to try out.                                                                                         |
| `set_non_tuneable_hyperparameters(non_tuneable_hyperparameter_choice)` | Reads in values for non-tuneable hyperparameters (i.e., doesn’t need to clog up the tuning output CSV).                                                                           | `non_tuneable_hyperparameter_choices` (`dict` of `str:int`): Non-tuneable hyperparameters that do not need to appear in the tuning output CSV.                                                  |
| `set_features(ningxiang_output)`                 | Reads in feature combinations for tuning.                                                                                                                                         | `ningxiang_output` (`dict` of `tuple:float`): Feature combinations for tuning.                                                                                                                  |
| `set_tuning_order(order)`                        | Sets the order of tuning for hyperparameters in JiaoCheng tuning.                                                                                                                 | `order` (`list`): Order of hyperparameters for tuning.                                                                                                                                          |
| `set_hyperparameter_default_values(default_values)` | Sets the default values for hyperparameters in JiaoCheng tuning.                                                                                                                  | `default_values` (`dict` of `str:int/float/str`): Default values for hyperparameters.                                                                                                           |
| `set_tuning_result_saving_address(address)`      | Sets the saving address for the tuning output CSV.                                                                                                                                | `address` (`str`): Does not need to include `.csv`.                                                                                                                                             |
| `tune(key_stats_only = False)`                   | Begins the tuning process.                                                                                                                                                        | `key_stats_only` (`bool`): If `True`, calculates only key statistics and skips non-important stats.                                                                                             |
| `read_in_tuning_result_df(address)`              | Reads in an existing DataFrame from a `.csv` file consisting of tuning results. Automatically populates the result array and checked array if CSV columns match parameter choices. | `address` (`str`): Must include `.csv`.                                                                                                                                                         |
| `set_tuning_best_model_saving_address(address)`  | Sets the address for exporting the best model as a pickle file.                                                                                                                   | `address` (`str`): Does not need to include `.pickle`.                                                                                                                                          |
| `view_best_combo_and_score()`                    | Views the current best combination and its validation score.                                                                                                                      | N/A                                                                                                                                                                                             |


### Attributes
| **Attribute**                                | **Type**            |
|----------------------------------------------|---------------------|
| `str:float`                                  | `dict`              |
| `non_tuneable_parameter_choices`             | `Dictionary`        |
| `checked`                                    | `np.array`          |
| `result`                                     | `np.array`          |
| `tuning_result_saving_address`               | `str`               |
| `best_model_saving_address`                  | `str`               |
| `best_score`                                 | `int`               |
| `best_combo`                                 | `list`              |
| `best_clf`                                   | `model object`      |
| `clf_type`                                   | `str`               |
| `combos`                                     | `List of lists`     |
| `n_items`                                    | `list`              |
| `hyperparameter_tuning_order`                | `list of hyperparameters` |
| `regression_extra_output_columns`            | `List (pre-set)`    |
| `classification_extra_output_columns`        | `list (pre-set)`    |


## JiaoChengB
>Ultra greedy hyperparameter tuning algorithm (Advanced)

**Background**

The purpose of this package is to provide a framework for feature-by-feature tuning, a different (and in most cases faster but less accurate) method compared to JiXi, YangZhou etc. Yet slightly more through (and hence expensive) than the original JiaoCheng algorithm.

Sometimes, a data scientist would be stuck in the midst of data cleaning, but would like to get a glimpse of how well this data is currently performing as a benchmark, and hence does not need to necessarily find the global maximum in the field space. JiaoCheng-B, being more greedy and hence training less combinations and taking less time than JiXi, is suitable for this purpose.

The package takes in X and y data for train, validate and test as DataFrame, as well as a dictionary of {hyperparameters name -> string: hyperparameter values as a list}, dictionary of default values for each hyperparameter and list of order of features, and autogenerates all combinations of these hyperparameters to be tuned. JiaoCheng-B has multiple stages, and each is one run on JiaoCheng: at the default values combination, and searches through different values of first hyperparameter whilst holding other hyperparameter values constant. The maximum combination from this search gets updated as the new ‘default value combination’ (now called ‘current max combination’) and the second hyperparameter is searched through holding other hyperparameter values of the ‘current max combination’ fixed. Once all hyperparameter have been searched through in this manner, if the ‘current max combination’ is the same as that before this round of all hyperparameter being searched, then this stage of the algorithm is completed. Else, another round of search is undertaken. At the completion of each stage, the order of the hyperparameters being searched is adjusted by moving the first hyperparameter to the back of the list order and repeating the stage with the new order (and default hp values once again). The algorithm only terminates when every hyperparameter had its turn as the first-searched hyperparameter

Due to the JX architecture memorising tuned combination from previous stages, the
algorithm typically has very similar total searched combinations as JiaoCheng[-A], and can
therefore be considered an insurance taken out for JiaoCheng[-A] at a very cheap premium.

### Methods
| **Method**                                        | **Description**                                                                                                                                                                  | **Parameters**                                                                                                                                                                                 |
|--------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `JiaoCheng()`                                    | Initialises the object.                                                                                                                                                          | N/A                                                                                                                                                                                             |
| `read_in_data(train_x, train_y, val_x, val_y, test_x, test_y)` | Reads in Train Test Split data.                                                                                                                                                  | `train_x` (`pd.DataFrame`): Training data features. <br> `train_y` (`pd.Series`): Training data labels. <br> `val_x` (`pd.DataFrame`): Validation data features. <br> `val_y` (`pd.Series`): Validation data labels. <br> `test_x` (`pd.DataFrame`): Test data features. <br> `test_y` (`pd.Series`): Test data labels. |
| `read_in_model(model, type, optimised_metric, pytorch_model)` | Reads in the underlying model class to tune for optimal parameters, the metric to optimise for, and whether the model is a PyTorch class model.                                  | `model` (Any model class): Must allow `.fit()` and `.predict()`. <br> `type` (`str`): Either "Classification" or "Regression". <br> `optimised_metric` (`str`): Metric to optimise, varies by `type`. <br> `pytorch_model` (`bool`): Whether the model is a PyTorch class model. |
| `set_hyperparameters(parameter_choices)`         | Reads in the different values of each hyperparameter to try. Automatically generates each combination.                                                                            | `parameter_choices` (`dict` of `str:list`): Hyperparameter names and their corresponding values to try.                                                                                         |
| `set_non_tuneable_hyperparameters(non_tuneable_hyperparameter_choice)` | Reads in values for non-tuneable hyperparameters.                                                                                                                                  | `non_tuneable_hyperparameter_choices` (`dict` of `str:int`): Non-tuneable hyperparameters that do not need to appear in the tuning output CSV.                                                  |
| `set_features(ningxiang_output)`                 | Reads in feature combinations for tuning.                                                                                                                                         | `ningxiang_output` (`dict` of `tuple:float`): Feature combinations for tuning.                                                                                                                  |
| `set_tuning_order(order)`                        | Sets the order of tuning for hyperparameters in JiaoCheng tuning.                                                                                                                 | `order` (`list`): Order of hyperparameters for tuning.                                                                                                                                          |
| `set_hyperparameter_default_values(default_values)` | Sets the default values for hyperparameters in JiaoCheng tuning.                                                                                                                  | `default_values` (`dict` of `str:int/float/str`): Default values for hyperparameters.                                                                                                           |
| `set_tuning_result_saving_address(address)`      | Sets the saving address for the tuning output CSV.                                                                                                                                | `address` (`str`): Does not need to include `.csv`.                                                                                                                                             |
| `tune(key_stats_only = False)`                   | Begins the tuning process.                                                                                                                                                        | `key_stats_only` (`bool`): If `True`, calculates only key statistics and skips non-important stats.                                                                                             |
| `read_in_tuning_result_df(address)`              | Reads in an existing DataFrame from a `.csv` file consisting of tuning results. Automatically populates the result array and checked array if CSV columns match parameter choices. | `address` (`str`): Must include `.csv`.                                                                                                                                                         |
| `set_tuning_best_model_saving_address(address)`  | Sets the address for exporting the best model as a pickle file.                                                                                                                   | `address` (`str`): Does not need to include `.pickle`.                                                                                                                                          |
| `view_best_combo_and_score()`                    | Views the current best combination and its validation score.                                                                                                                      | N/A                                                                                                                                                                                             |

### Attributes

| **Attribute**                                | **Type**            |
|----------------------------------------------|---------------------|
| `np.array`                                   | `np.array`          |
| `result`                                     | `np.array`          |
| `tuning_result_saving_address`               | `str`               |
| `best_model_saving_address`                  | `str`               |
| `best_score = -np.inf`                       | `int`               |
| `best_combo`                                 | `list`              |
| `best_clf`                                   | `model object`      |
| `clf_type`                                   | `str` – 'Regression' or 'Classification' |
| `combos`                                     | `List of lists`     |
| `n_items`                                    | `list` - Denotes how many values in each hyperparameter dimension |
| `hyperparameter_tuning_order`                | `list of hyperparameters` |
| `regression_extra_output_columns`            | `List (pre-set)`    |
| `classification_extra_output_columns`        | `list (pre-set)`    |


## YangZhouB
>Greedy hyperparameter tuning algorithm

**Background**

**YangZhou-B** begins by searching all **Cruise Combinations** (mathematical combinations of all cruise indices from each dimension). Cruise indices are:

- `[0, 4]`, `[0, 5]`, `[0, 3, 6]` or `[0, 4, 7]` for dimensions containing 5, 6, 7, and 8 values respectively.

The maximum gap between two indices is 5, and the minimum is 3.

Then, starting with the median combination (median index of each dimension) as the initial core, the **Guidance Algorithm** is activated, in which all the horizontal/vertical neighboring combinations are searched (i.e., all the combinations which are the same as the core except for one dimension being +1 or -1 compared to previously). If `score(neighbour) - score(core) >= -0.005`, then the neighboring combination is added as the new core.

The **Guidance Algorithm** is then repeated for each of the new cores. When no new cores need to be tested, the maximum scoring combination will have all surrounding neighbors searched, and if a new maximum scoring combination is found, then it will also get its neighbors searched until no new maximum scoring combination can be found. The **Guidance Algorithm** is then terminated.

The **Cruise Algorithm** is then subsequently activated, in which each of the cruise combinations' scores will be compared to the current best scoring combination and its surrounding +/-1 neighbor block (including itself). If a cruise combination’s score is higher than the:

`warning threshold = mean(best surrounding block) - qt(0.95) * (sd / √(3d))`

then the **Guidance Algorithm** will be restarted on that cruise combination.

The **Cruise Algorithm** terminates once all cruise combinations have been compared to the warning threshold (which could change as the **Cruise Algorithm** goes on).

Once the **Cruise Algorithm** ends, the **Guidance Algorithm** gets activated one more time starting at the current maximum scoring combination, and the whole **YangZhou-B Algorithm** ends when this call of **Guidance Algorithm** is finished.

> **Note**: Although scores of certain combinations will undoubtedly be called upon multiple times, they can be stored and thus the expensive basic operation of train-searching a combination will only ever need to be completed once for each combination.


***Algorithm Assumptions***
1. The scores observed from the same {data, model, hyperparameter combination, split size}
belongs to the same underlying population which are normally distributed around a theoretical
value.
i.e. Accuracies of SVM on a fixed set of hyperparameters with 80-20 split size on the same set
of data (but with random holdouts of 80% training data) is considered to be sampled from the
same population (and thereby same distribution)

### Methods
| **Method**                                        | **Description**                                                                                                                                                                  | **Parameters**                                                                                                                                                                                 |
|--------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `YangZhouB()`                                    | Initialises the object.                                                                                                                                                          | N/A                                                                                                                                                                                             |
| `read_in_data(train_x, train_y, val_x, val_y, test_x, test_y)` | Reads in Train Test Split data.                                                                                                                                                  | `train_x` (`pd.DataFrame`): Training data features. <br> `train_y` (`pd.Series`): Training data labels. <br> `val_x` (`pd.DataFrame`): Validation data features. <br> `val_y` (`pd.Series`): Validation data labels. <br> `test_x` (`pd.DataFrame`): Test data features. <br> `test_y` (`pd.Series`): Test data labels. |
| `read_in_model(model, type, optimised_metric, pytorch_model)` | Reads in the underlying model class to tune for optimal parameters, the metric to optimise for, and whether the model is a PyTorch class model.                                  | `model` (Any model class): Must allow `.fit()` and `.predict()`. <br> `type` (`str`): Either "Classification" or "Regression". <br> `optimised_metric` (`str`): Metric to optimise, varies by `type`. <br> `pytorch_model` (`bool`): Whether the model is a PyTorch class model. |
| `set_hyperparameters(parameter_choices)`         | Reads in the different values of each hyperparameter to try. Automatically generates each combination.                                                                            | `parameter_choices` (`dict` of `str:list`): Hyperparameter names and their corresponding values to try.                                                                                         |
| `set_non_tuneable_hyperparameters(non_tuneable_hyperparameter_choice)` | Reads in values for non-tuneable hyperparameters.                                                                                                                                  | `non_tuneable_hyperparameter_choices` (`dict` of `str:int`): Non-tuneable hyperparameters that do not need to appear in the tuning output CSV.                                                  |
| `set_features(ningxiang_output)`                 | Reads in feature combinations for tuning.                                                                                                                                         | `ningxiang_output` (`dict` of `tuple:float`): Feature combinations for tuning.                                                                                                                  |
| `set_tuning_result_saving_address(address)`      | Sets the saving address for the tuning output CSV.                                                                                                                                | `address` (`str`): Does not need to include `.csv`.                                                                                                                                             |
| `tune(key_stats_only = False)`                   | Begins the tuning process.                                                                                                                                                        | `key_stats_only` (`bool`): If `True`, calculates only key statistics and skips non-important stats.                                                                                             |
| `tune_parallel(part, splits, key_stats_only = False)` | Begins the tuning process, splitting all combinations into parts and tuning the specified part.                                                                                   | `part` (`int`): The specific part to tune. <br> `splits` (`int`): The number of splits. <br> `key_stats_only` (`bool`): If `True`, calculates only key statistics and skips non-important stats. |
| `read_in_tuning_result_df(address)`              | Reads in an existing DataFrame from a `.csv` file consisting of tuning results. Automatically populates the result array and checked array if CSV columns match parameter choices. | `address` (`str`): Must include `.csv`.                                                                                                                                                         |
| `set_tuning_best_model_saving_address(address)`  | Sets the address for exporting the best model as a pickle file.                                                                                                                   | `address` (`str`): Does not need to include `.pickle`.                                                                                                                                          |
| `view_best_combo_and_score()`                    | Views the current best combination and its validation score.                                                                                                                      | N/A                                                                                                                                                                                             |

### Attributes
| **Attribute**                                | **Type**            |
|----------------------------------------------|---------------------|
| `hyperparameters`                            | `list`              |
| `feature_n_ningxiang_score_dict`             | `Dictionary`        |
| `str:float`                                  | `str:float`         |
| `non_tuneable_parameter_choices`             | `Dictionary`        |
| `str:str/float/int`                          | `str:str/float/int` |
| `checked`                                    | `np.array`          |
| `result`                                     | `np.array`          |
| `checked_core`                               | `np.array`          |
| `been_cruised`                               | `np.array`          |
| `combo`                                      | `np.array`          |
| `been_best`                                  | `np.array`          |
| `tuning_result_saving_address`               | `str`               |
| `best_model_saving_address`                  | `str`               |
| `best_score = -np.inf`                       | `int`               |
| `best_combo`                                 | `list`              |
| `best_clf`                                   | `model object`      |
| `clf_type`                                   | `str` – 'Regression' or 'Classification' |
| `n_items`                                    | `list` - Denotes how many values in each hyperparameter dimension |
| `regression_extra_output_columns`            | `List (pre-set)`    |
| `classification_extra_output_columns`        | `list (pre-set)`    |



# *Usage Examples*
## *Create Dataset*
```python
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

# create regression data
X_reg, y_reg = make_regression(
    n_samples=100, n_features=5, noise=0.1, random_state=42
)
X_reg_df = pd.DataFrame(
    X_reg, columns=[f"feature_{i+1}" for i in range(X_reg.shape[1])]
)
y_reg_series = pd.Series(y_reg, name="target")

## train-test split
X_reg_df_train, X_reg_df_valtest, y_reg_series_train, y_reg_series_valtest = (
    train_test_split(X_reg_df, y_reg_series, test_size=0.3, random_state=42)
)
X_reg_df_val, X_reg_df_test, y_reg_series_val, y_reg_series_test = train_test_split(
    X_reg_df_valtest, y_reg_series_valtest, test_size=0.5, random_state=42
)

# Create classification data
X_class_2, y_class_2 = make_classification(
    n_samples=100,
    n_features=5,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=42,
)
X_class_2_df = pd.DataFrame(
    X_class_2, columns=[f"feature_{i+1}" for i in range(X_class_2.shape[1])]
)
y_class_2_series = pd.Series(y_class_2, name="target")

## train-test split
(
    X_class_2_df_train,
    X_class_2_df_valtest,
    y_class_2_series_train,
    y_class_2_series_valtest,
) = train_test_split(X_class_2_df, y_class_2_series, test_size=0.3, random_state=42)
X_class_2_df_val, X_class_2_df_test, y_class_2_series_val, y_class_2_series_test = (
    train_test_split(
        X_class_2_df_valtest,
        y_class_2_series_valtest,
        test_size=0.5,
        random_state=42,
    )
)
```

## *NingXiang Example*
*RF features*
```python
# REGRESSION
feature_selector = NingXiang()
feature_selector.read_in_train_data(X_reg_df_train, y_reg_series_test)
feature_selector.set_model_type('Regression')

## feature order object
reg_feature_order_dict = feature_selector.get_rf_based_feature_combinations(n_jobs=-1)

feature_selector.show_rf_stats()


# CLASSIFICATION
feature_selector = NingXiang()
feature_selector.read_in_train_data(X_class_2_df_train, y_class_2_series_train)
feature_selector.set_model_type('Classification')

## feature order object
class_feature_order_dict = feature_selector.get_rf_based_feature_combinations(n_jobs=-1)

feature_selector.show_rf_stats()
```

## *JiaoCheng Example*
```python
# CLASSIFICATION, WITH FEATURES TUNED AS HYPERPARAMETER

from JXAutoML.JiaoCheng import JiaoCheng as tuner
from sklearn.ensemble import RandomForestClassifier as clf

# what values to try for each hyperparameter
parameter_choices = {
    "max_depth": (3, 6, 12, 24),
    "max_samples": (0.4, 0.55, 0.7, 0.85),
}

# what values to set non-tuneable parameters/hyperparameters
non_tunable_hyperparameters_dict = {"random_state": 42, "n_jobs": -1}

tuning_order = ["features", "max_depth", "max_samples"]
default_hyperparameter_values = {"features": 0, "max_depth": 3, "max_samples": 0.4}

tuner = tuner()

# define what model we are tuning
tuner.read_in_model(
    clf, 'Classification', pytorch_model=False, optimised_metric='accuracy'
)

# read in the data for training and validation
tuner.read_in_data(X_class_2_df_train, y_class_2_series_train, X_class_2_df_val, y_class_2_series_val, X_class_2_df_test, y_class_2_series_test)

# set what hp values to tune
tuner.set_hyperparameters(parameter_choices)
# WARNING: this may take a while if no. tuneable hyperparameters are large

# set up hp values that need to be changed from default but NOT to be tuned
tuner.set_non_tuneable_hyperparameters(non_tunable_hyperparameters_dict)

# set up feature importance ordering
tuner.set_features(class_feature_order_dict)
# WARNING: this may take a while if no. tuneable hyperparameters are large

# set up the order of hyperparameters when iteratively tuning using JiaoCheng
tuner.set_tuning_order(tuning_order)

# set up the default hp values for first iteration of tuning JiaoCheng
tuner.set_hyperparameter_default_values(default_hyperparameter_values)

# set up where to save the tuning result csv
tuner.set_tuning_result_saving_address(
    'jiaocheng_test_tuning_result.csv'
)

# set up where to save the current best model
tuner.set_best_model_saving_address(
    'jiaocheng_test_tuning_best_model'
)

tuner.tune()
```

## *JiaoChengB Example*
```python
# REGRESSION
from JXAutoML.JiaoChengB import JiaoChengB as tuner
from sklearn.ensemble import RandomForestRegressor as clf

# what values to try for each hyperparameter
parameter_choices = {
    "max_depth": (3, 6, 12, 24),
    "max_samples": (0.4, 0.55, 0.7, 0.85),
}

# what values to set non-tuneable parameters/hyperparameters
non_tunable_hyperparameters_dict = {"random_state": 42, "n_jobs": -1}

tuning_order = ["max_depth", "max_samples"]
default_hyperparameter_values = {"max_depth": 3, "max_samples": 0.4}


tuner = tuner()

# define what model we are tuning
tuner.read_in_model(
    clf, 'Regression', pytorch_model=False, optimised_metric='r2'
)

# read in the data for training and validation
tuner.read_in_data(X_reg_df_train, y_reg_series_train, X_reg_df_val, y_reg_series_val, X_reg_df_test, y_reg_series_test)

# set what hp values to tune
tuner.set_hyperparameters(parameter_choices)
# WARNING: this may take a while if no. tuneable hyperparameters are large

# set up hp values that need to be changed from default but NOT to be tuned
tuner.set_non_tuneable_hyperparameters(non_tunable_hyperparameters_dict)

# set up the order of hyperparameters when iteratively tuning using JiaoCheng
tuner.set_tuning_order(tuning_order)

# set up the default hp values for first iteration of tuning JiaoCheng
tuner.set_hyperparameter_default_values(default_hyperparameter_values)

# set up where to save the tuning result csv
tuner.set_tuning_result_saving_address(
    'jiaochengb_test_tuning_result.csv'
)

# set up where to save the current best model
tuner.set_best_model_saving_address(
    'jiaochengb_test_tuning_best_model'
)

tuner.tune()
```

## *YangZhou Example*
```python
# CLASSIFICATION, WITH FEATURES TUNED AS HYPERPARAMETER
from JXAutoML.YangZhouB import YangZhouB as tuner
from sklearn.ensemble import RandomForestClassifier as clf

# what values to try for each hyperparameter
parameter_choices = {
    "max_depth": (3, 6, 12, 24),
    "max_samples": (0.4, 0.55, 0.7, 0.85),
}

# what values to set non-tuneable parameters/hyperparameters
non_tunable_hyperparameters_dict = {"random_state": 42, "n_jobs": -1}

tuner = tuner()

# define what model we are tuning
tuner.read_in_model(
    clf, 'Classification', pytorch_model=False, optimised_metric='accuracy'
)

# read in the data for training and validation
tuner.read_in_data(X_class_2_df_train, y_class_2_series_train, X_class_2_df_val, y_class_2_series_val, X_class_2_df_test, y_class_2_series_test)

# set what hp values to tune
tuner.set_hyperparameters(parameter_choices)
# WARNING: this may take a while if no. tuneable hyperparameters are large

# set up hp values that need to be changed from default but NOT to be tuned
tuner.set_non_tuneable_hyperparameters(non_tunable_hyperparameters_dict)

# set up feature importance ordering

tuner.set_features(class_feature_order_dict)
# WARNING: this may take a while if no. tuneable hyperparameters are large

# set up where to save the tuning result csv
tuner.set_tuning_result_saving_address(
    "yangzhou_test_tuning_result.csv"
)

# set up where to save the current best model
tuner.set_best_model_saving_address(
    "yangzhou_test_tuning_best_model"
)

tuner.tune()
```