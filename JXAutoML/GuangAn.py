import pandas as pd
import numpy as np
import copy
import statsmodels.api as sm
import pickle
import time
from collections import defaultdict as dd

from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, average_precision_score, roc_auc_score


class GuangAn:

    def __init__(self):
        """ Initialise class """
        self._initialise_objects()

        print('GuangAn Initialised')

    def _initialise_objects(self):
        """ Helper to initialise objects """

        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.test_x = None
        self.test_y = None
        self.checked_dict = None
        self.model = None
        self.parameter_ranges = None
        self.hyperparameters = None
        self.tuning_result_saving_address = None
        self._up_to = 0
        self._seed = 19040822
        self.best_score = -np.inf
        self.best_combo = None
        self.best_clf = None
        self.clf_type = None
        self._tune_features = False
        self.non_tuneable_parameter_choices = list()
        self.categorical = None
        self.transform = None
        self._is_new_best = 0
        self.best_model_saving_address = None
        self._feature_combo_n_index_map = None
        self.pytorch_model = False
        self.optimised_metric = False

        self.regression_extra_output_columns = ['Train r2', 'Val r2', 'Test r2',
                                                'Train rmse', 'Val rmse', 'Test rmse', 'Train mape', 'Val mape', 'Test mape', 'Time']
        self.classification_extra_output_columns = ['Train accuracy', 'Val accuracy', 'Test accuracy',
                                                    'Train balanced_accuracy', 'Val balanced_accuracy', 'Test balanced_accuracy', 'Train f1', 'Val f1', 'Test f1',
                                                    'Train precision', 'Val precision', 'Test precision', 'Train recall', 'Val recall', 'Test recall', 'Time']

    def read_in_data(self, train_x, train_y, val_x, val_y, test_x, test_y):
        """ Reads in train validate test data for tuning """

        self.train_x = train_x
        print("Read in Train X data")

        self.train_y = train_y
        print("Read in Train y data")

        self.val_x = val_x
        print("Read in Val X data")

        self.val_y = val_y
        print("Read in Val y data")

        self.test_x = test_x
        print("Read in Test X data")

        self.test_y = test_y
        print("Read in Test y data")

    def read_in_model(self, model, type, optimised_metric=None, pytorch_model=False):
        """ Reads in underlying model object for tuning, and also read in what type of model it is """

        assert type == 'Classification' or type == 'Regression'  # check

        self.clf_type = type

        if self.clf_type == 'Classification':
            assert optimised_metric in [None, 'accuracy', 'f1', 'precision',
                                        'recall', 'balanced_accuracy', 'AP', 'AUC'], "evaluation_metric for classification must be one of ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy', 'AP', 'AUC']"
        if self.clf_type == 'Regression':
            assert optimised_metric in [
                None, 'r2', 'rmse', 'mape'], "evaluation_metric for regression must be one of ['r2', 'rmse', 'mape']"

        if self.clf_type == 'Classification':
            self.optimised_metric = 'accuracy' if optimised_metric is None else optimised_metric
        elif self.clf_type == 'Regression':
            self.optimised_metric = 'r2' if optimised_metric is None else optimised_metric

        # record
        self.model = model

        self.pytorch_model = pytorch_model

        print(
            f'Successfully read in model {self.model}, which is a {self.clf_type} model optimising for {self.optimised_metric}')

    def set_hyperparameters(self, parameter_ranges_orig):
        """ Input hyperparameter choices """

        self.parameter_ranges = parameter_ranges_orig
        self._sort_hyperparameter_ranges()

        self.hyperparameters = list(self.parameter_ranges.keys())

        self.categorical = {hp: False for hp in self.hyperparameters}
        self.transform = {hp: False for hp in self.hyperparameters}

        # automatically setup checked dictionary and tuning dataframe
        self._get_checked_dict()
        self._setup_tuning_result_df()

        # automatically calculate the original bounds
        self.original_bounds = [(self.parameter_ranges[key], key)
                                for key in self.parameter_ranges]

        print("Successfully recorded hyperparameter choices")

    def _sort_hyperparameter_ranges(self):
        """ Helper to ensure all hyperparameter choice values are in order from lowest to highest """

        for key in self.parameter_ranges:
            tmp = copy.deepcopy(list(self.parameter_ranges[key]))
            tmp = self._sort_with_none(tmp)
            if type(self.parameter_ranges[key]) is set:
                self.parameter_ranges[key] = set(tmp)
            else:
                self.parameter_ranges[key] = tuple(tmp)

    def _sort_with_none(self, lst):
        """ Helper to sort hyperparameters with None values """
        if None in lst:
            no_none_list = [i for i in lst if i is not None]
            no_none_list.sort()
            no_none_list = [None]+no_none_list
            return no_none_list
        lst.sort()
        return lst

    def _setup_tuning_result_df(self):
        """ Helper to set up tuning result dataframe """

        tune_result_columns = copy.deepcopy(self.hyperparameters)

        if self._tune_features:
            tune_result_columns.append('feature combo ningxiang score')

        # Different set of metric columns for different types of models
        if self.clf_type == 'Classification':
            tune_result_columns.extend(
                self.classification_extra_output_columns)
        elif self.clf_type == 'Regression':
            tune_result_columns.extend(self.regression_extra_output_columns)

        self.tuning_result = pd.DataFrame(
            {col: list() for col in tune_result_columns})

    def _get_checked_dict(self):
        """ Helper to set up checked list """

        self.checked_dict = dict()

    def set_non_tuneable_hyperparameters(self, non_tuneable_hyperparameter_choice):
        """ Input Non tuneable hyperparameter choice """

        if type(non_tuneable_hyperparameter_choice) is not dict:
            raise TypeError(
                'non_tuneable_hyeprparameters_choice must be dict, please try again')

        for nthp in non_tuneable_hyperparameter_choice:
            if type(non_tuneable_hyperparameter_choice[nthp]) in (set, list, tuple, dict):
                raise TypeError(
                    'non_tuneable_hyperparameters_choice must not be of array-like type')

        self.non_tuneable_parameter_choices = non_tuneable_hyperparameter_choice

        print("Successfully recorded non_tuneable_hyperparameter choices")

    def read_in_transform(self, transform_update):
        """ Function to read in transformation settings """

        if not self.hyperparameters:
            raise AttributeError(
                "Missing hyperparameter choices, please run .set_hyperparameters() first")

        if type(transform_update) is not dict:
            raise TypeError(
                'transform_update should be a dict, please re-enter')

        for key in transform_update:
            self.transform[key] = transform_update[key]

        print('Updated transform dictionary:', self.transform)

    def read_in_categorical(self, categorical_update):
        """ Function to read in categorical settings """

        if not self.hyperparameters:
            raise AttributeError(
                "Missing hyperparameter choices, please run .set_hyperparameters() first")

        if type(categorical_update) is not list:
            raise TypeError(
                'categorical_update should be a list, please re-enter')

        for key in categorical_update:
            self.categorical[key] = True

            self.parameter_ranges[key] = {
                'values': tuple(self.parameter_ranges[key])}

        self.original_bounds = [(self.parameter_ranges[key], key)
                                for key in self.parameter_ranges]

        print('Updated categorical dictionary:', self.categorical)
        print('Updated original bounds dict:', self.original_bounds)

    def set_features(self, ningxiang_output):
        """ Input features """

        if type(ningxiang_output) is not dict:
            raise TypeError("Please ensure NingXiang output is a dict")

        if not self.hyperparameters:
            raise AttributeError(
                "Missing hyperparameter choices, please run .set_hyperparameters() first")

        for feature in list(ningxiang_output.keys())[-1]:
            if feature not in list(self.train_x.columns):
                raise ValueError(
                    f'feature {feature} in ningxiang output is not in train_x. Please try again')

            if feature not in list(self.val_x.columns):
                raise ValueError(
                    f'feature {feature} in ningxiang output is not in val_x. Please try again')

            if feature not in list(self.test_x.columns):
                raise ValueError(
                    f'feature {feature} in ningxiang output is not in test_x. Please try again')

        # sort ningxiang just for safety, and store up
        ningxiang_output_sorted = self._sort_features(ningxiang_output)
        self.feature_n_ningxiang_score_dict = ningxiang_output_sorted

        # activate this switch
        self._tune_features = True

        # update previous internal structures based on first set of hyperparameter choices
        # here used numbers instead of tuples as the values in parameter_choices; thus need another mapping to get map back to the features
        self.parameter_ranges['features'] = set(
            [i for i in range(len(ningxiang_output_sorted))])
        self._feature_combo_n_index_map = {i: list(ningxiang_output_sorted.keys())[
            i] for i in range(len(ningxiang_output_sorted))}

        self.hyperparameters = list(self.parameter_ranges.keys())

        self.categorical['features'] = False
        self.transform['features'] = False

        # automatically setup checked dictionary and tuning dataframe
        self._get_checked_dict()
        self._setup_tuning_result_df()

        # automatically calculate the original bounds
        self.original_bounds = [(self.parameter_ranges[key], key)
                                for key in self.parameter_ranges]

        print("Successfully recorded tuneable feature combination choices and updated relevant internal structures")

    def _sort_features(self, ningxiang_output):
        """ Helper for sorting features based on NingXiang values (input dict output dict) """

        ningxiang_output_list = [(key, ningxiang_output[key])
                                 for key in ningxiang_output]

        ningxiang_output_list.sort(key=lambda x: x[1])

        ningxiang_output_sorted = {x[0]: x[1] for x in ningxiang_output_list}

        return ningxiang_output_sorted

    def _get_coords_from_bounds(self, bounds):
        """ Function to get initial coordinates to tune """

        # Setup the initial list that is used in classic JiaXing combo getting algorithm
        if type(bounds[0][0]) is tuple:
            boundary_coordinates = [[bounds[0][0][i]] for i in range(2)]
        elif type(bounds[0][0]) is set:
            if len(bounds[0][0]) == 1:
                boundary_coordinates = [[list(bounds[0][0])[0]]]
            else:
                boundary_coordinates = [[min(list(bounds[0][0]))], [
                    max(list(bounds[0][0]))]]
        elif type(bounds[0][0]) is dict:
            boundary_coordinates = [[bounds[0][0]['values'][i]]
                                    for i in range(len(bounds[0][0]['values']))]

        # Second part of classic JiaXing combo getting algorithm
        for i in range(1, len(bounds)):
            old_boundary_coordinates = copy.deepcopy(boundary_coordinates)
            boundary_coordinates = list()

            values = bounds[i]

            for init_coord in old_boundary_coordinates:
                if type(values[0]) is tuple:  # tuple: continuous values
                    for value in values[0]:
                        tmp = copy.deepcopy(init_coord)
                        tmp.append(value)
                        boundary_coordinates.append(tmp)

                # set: semi-continuous values (ordinal or floats but not continuous)
                elif type(values[0]) is set:
                    if len(values[0]) == 1:
                        tmp = copy.deepcopy(init_coord)
                        tmp.append(list(values[0])[0])
                        boundary_coordinates.append(tmp)
                    else:
                        for value in [min(list(values[0])), max(list(values[0]))]:
                            tmp = copy.deepcopy(init_coord)
                            tmp.append(value)
                            boundary_coordinates.append(tmp)

                elif type(values[0]) is dict:  # dict: discrete values
                    for value in values[0]['values']:
                        tmp = copy.deepcopy(init_coord)
                        tmp.append(value)
                        boundary_coordinates.append(tmp)

        return boundary_coordinates

    def _get_centre_components(self, bounds, categorical):
        """ Helper that gets the centres of a bound as lists (considering for categorical) """

        # Classic JiaXing getting combo algorithm
        centre_components = list()
        tmp_cat = copy.deepcopy(categorical)
        for i in range(len(bounds)):

            # tuple: continuous values
            if type(bounds[i][0]) is tuple or type(bounds[i][0]) is list:
                # take the mean value
                centre_components.append(sum(bounds[i][0])/2)

            # set: semi-continuous values (ordinal or floats but not continuous)
            elif type(bounds[i][0]) is set:
                n_semicont_values = len(bounds[i][0])
                if n_semicont_values <= 2:

                    # just input the tuple(set) as the centre (which will be recognised as two discrete)
                    centre_components.append(tuple(bounds[i][0]))

                    # set categorical of this variable to true because no more semi-continuous values in between the current bounds
                    tmp_cat[self.hyperparameters[i]] = True

                else:
                    # take the middle
                    components_list = list(bounds[i][0])
                    components_list.sort()
                    centre_components.append(
                        components_list[n_semicont_values//2])

            elif type(bounds[i][0]) is dict:  # dict: discrete values
                # input the value (a set) which will be recognised as discrete
                centre_components.append(bounds[i][0]['values'])

        # returns 1. components that can be unpacked into multiple centres; 2. new categorical labels
        return centre_components, tmp_cat

    def _unpack_centre(self, centre_components):
        """ Helper to unpack centre components into centre """

        # Classic JiaXing algorithm for getting all combinations
        centres = [[]]
        for i in range(len(centre_components)):
            old_centres = copy.deepcopy(centres)
            centres = list()
            if type(centre_components[i]) is tuple:
                for obj in centre_components[i]:
                    for cent in old_centres:
                        tmp_cent = copy.deepcopy(cent)
                        tmp_cent.append(obj)
                        centres.append(tmp_cent)
            else:
                for cent in old_centres:
                    tmp_cent = copy.deepcopy(cent)
                    tmp_cent.append(centre_components[i])
                    centres.append(tmp_cent)

        return [tuple(centre) for centre in centres]

    def _get_categorical(self, new_cat, boundaries):
        """ Helper to get all combos of categorical feature's values (for use in OLS) """

        # Classic JiaXing algorithm for getting all combinations
        out = [[]]
        for hyperparameter in new_cat:
            if new_cat[hyperparameter] is True:
                old_out = copy.deepcopy(out)
                out = list()

                val_list = list(boundaries[hyperparameter])
                val_list_unique = list(set(val_list))
                val_list_unique.sort()

                for val in val_list_unique:
                    for lst in old_out:
                        tmp = copy.deepcopy(lst)
                        tmp.append(val)
                        out.append(tmp)

        return out

    def _get_new_bounds(self, bounds, centre, categorical):
        """ Function to get new bounds """

        # get the range components that make the 2^d bounds
        range_components = self._get_range_components(
            bounds, centre, categorical)

        # make the new bounds from components
        new_bounds = self._make_bounds(range_components)

        return new_bounds

    def _get_range_components(self, bounds, centre, categorical):
        """ Helper that gets the range components """

        range_components = list()
        for i in range(len(bounds)):

            if type(bounds[i][0]) is tuple:
                lower_range = (bounds[i][0][0], centre[i])
                upper_range = (centre[i], bounds[i][0][1])

                ranges = (lower_range, upper_range)

                range_components.append((ranges, bounds[i][1]))

            elif type(bounds[i][0]) is set:

                if categorical[self.hyperparameters[i]] is False:
                    lower_range_min_max = (min(list(bounds[i][0])), centre[i])
                    upper_range_min_max = (centre[i], max((bounds[i][0])))
                    lower_range = list()
                    for orig_val in list(self.parameter_ranges[self.hyperparameters[i]]):
                        if orig_val <= lower_range_min_max[1] and lower_range_min_max[0] <= orig_val:
                            lower_range.append(orig_val)
                    lower_range.sort()
                    lower_range = set(lower_range)

                    upper_range = list()
                    for orig_val in list(self.parameter_ranges[self.hyperparameters[i]]):
                        if orig_val <= upper_range_min_max[1] and upper_range_min_max[0] <= orig_val:
                            upper_range.append(orig_val)
                    upper_range.sort()
                    upper_range = set(upper_range)

                    ranges = (lower_range, upper_range)

                    range_components.append((ranges, bounds[i][1]))

                else:
                    ranges = bounds[i][0]
                    range_components.append((ranges, bounds[i][1]))

            elif type(bounds[i][0]) is dict:

                ranges = set(bounds[i][0]['values'])

                range_components.append((ranges, bounds[i][1]))

        return range_components

    def _make_bounds(self, range_components, min_threshold=0.1):
        """ Helper that makes the bounds using range components """

        reach_threshold_tuple = 0  # set checks for reaching a minimum threshold
        total_tuple = 0

        # Algorithm to create all bounds
        if type(range_components[0][0]) is tuple:
            bounds = [[(range_components[0][0][i], range_components[0][1])]
                      for i in range(2)]  # hardcode cos bounds can only have 2 values

            if type(range_components[0][0][0]) is tuple:
                total_tuple += 1
                if range_components[0][0][0][1] - range_components[0][0][0][0] <= min_threshold:
                    reach_threshold_tuple += 1

        elif type(range_components[0][0]) is set:
            tmp_tup = tuple(range_components[0][0])
            bounds = [[({tmp_tup[i]}, range_components[0][1])]
                      for i in range(len(tmp_tup))]

        for i in range(1, len(range_components)):
            old_bounds = copy.deepcopy(bounds)
            bounds = list()

            values = range_components[i]

            for bound in old_bounds:

                if type(values[0]) == tuple:
                    for value in values[0]:
                        tmp = copy.deepcopy(bound)
                        tmp.append((value, values[1]))

                        bounds.append(tmp)

                    if type(values[0][0]) is tuple:
                        total_tuple += 1
                        if values[0][0][1] - values[0][0][0] <= min_threshold:
                            reach_threshold_tuple += 1

                elif type(values[0]) == set:
                    for value in list(values[0]):
                        tmp = copy.deepcopy(bound)
                        tmp.append(({value}, values[1]))

                        bounds.append(tmp)

        if total_tuple and reach_threshold_tuple == total_tuple:  # if total != 0 and all reached threshold
            return False

        return bounds

    def _rebuild_bounds_to_original_format(self, tmp_boundary, new_cat):
        """ Helper to rebuild current format of bounds (as a df) into original bound format"""

        tmp_boundary = tmp_boundary.drop(['score'], axis=1)

        bounds_original_format = list()
        for col in tmp_boundary.columns:

            # if already categorical: just keep it as categorical
            if new_cat[col] == True:
                bounds_original_format.append(
                    ({'values': tuple(set(tmp_boundary[col]))}, col))

            else:
                col_vals = list(set(tmp_boundary[col]))

                if type(self.parameter_ranges[col]) is set:
                    tmp = list()
                    curr_val_max = max(col_vals)
                    curr_val_min = min(col_vals)
                    for orig_val in list(self.parameter_ranges[col]):
                        if orig_val <= curr_val_max and curr_val_min <= orig_val:
                            tmp.append(orig_val)
                    tmp.sort()
                    tmp = set(tmp)
                    bounds_original_format.append((tmp, col))

                else:  # continuous values
                    bounds_original_format.append(
                        ((min(col_vals), max(col_vals)), col))

        return bounds_original_format

    def _get_list_from_df(self, df):
        """ Helper to get df rows into list form """

        out = list()
        for row in df.iterrows():
            out.append(list(row[1].values))

        return out

    def _get_protective_bounds(self, bounds):
        """ Helper to get protective bounds  - for boundaries"""

        protective_bounds = list()
        for bound in bounds:
            if type(bound[0]) is dict:  # categorical values become a set
                protective_bounds.append(set(bound[0]['values']))
            elif type(bound[0]) is set:  # semi_categorical values become tuple
                protective_bounds.append(
                    (min(list(bound[0])), max(list(bound[0]))))
            elif type(bound[0]) is tuple:  # continuous values stay as tuple
                protective_bounds.append(bound[0])

        return protective_bounds

    def _get_protective_bounds2(self, tmp_boundary, new_cat):
        """ Helper to get protective bounds - for centres """

        protective_bounds = list()
        for col in tmp_boundary.columns:
            if col == 'score':
                continue

            col_values = list(tmp_boundary[col])

            if new_cat[col]:  # categorical values - only one left - set
                protective_bounds.append({col_values[0], })

            else:  # continuous values - tuple
                protective_bounds.append([min(col_values), max(col_values)])

        return protective_bounds

    def _in_protective_bounds(self, centre):
        """ Determine whether centre is in protective_bounds """

        for i in range(len(centre)):
            if type(self._protective_bounds[i]) is set:  # categorical
                # not matching any of the categorical values
                if centre[i] not in self._protective_bounds[i]:
                    return False
            else:  # continuous values
                # outside of boundary tuple
                if centre[i] > self._protective_bounds[i][1] or centre[i] < self._protective_bounds[i][0]:
                    return False

        return True

    def _protective_bounds_to_original_bounds(self):
        """ Helper to turn protective bounds back to original bounds """

        # TODO: Potential problem in this function

        protective_to_original_bounds = list()
        for i in range(len(self._protective_bounds)):
            if type(self._protective_bounds[i]) is set:  # discrete
                protective_to_original_bounds.append(
                    ({'values': tuple(self._protective_bounds[i])}, self.original_bounds[i][1]))
            else:  # semicont or disc
                if type(self.original_bounds[i][0]) is set:

                    tmp = list()
                    for val in self.original_bounds[i][0]:
                        if val >= self._protective_bounds[i][0] and val <= self._protective_bounds[i][1]:
                            tmp.append(val)

                    protective_to_original_bounds.append(
                        (set(tmp), self.original_bounds[i][1]))
                else:
                    protective_to_original_bounds.append(
                        (tuple(self._protective_bounds[i]), self.original_bounds[i][1]))

        return protective_to_original_bounds

    def tune(self, key_stats_only=False):
        """ Begin tuning """

        if self.train_x is None or self.train_y is None or self.val_x is None or self.val_y is None or self.test_x is None or self.test_y is None:
            raise AttributeError(
                " Missing one of the datasets, please run .read_in_data() ")

        if self.model is None:
            raise AttributeError(
                " Missing model, please run .read_in_model() ")

        if self.tuning_result_saving_address is None:
            raise AttributeError(
                "Missing tuning result csv saving address, please run .set_tuning_result_saving_address() first")

        print("Begin Guidance")

        self.key_stats_only = key_stats_only

        self._round = 0

        # start by putting original bounds into a list; this list is the object that will control whether algorithm has terminated
        bounds_list = [self.original_bounds]

        while bounds_list:  # get reset every iteration, so algo will keep running if there are bounds to operate on
            print("Round:", self._round)

            old_bounds_list = copy.deepcopy(bounds_list)
            tmp_bounds_list = list()

            for k in range(len(old_bounds_list)):  # now run algorithm on every bound

                # get the coordinates that define the bounds
                coords_to_tune = self._get_coords_from_bounds(
                    old_bounds_list[k])

                # get all coordinates into a DataFrame - used for getting boundary
                boundaries = pd.DataFrame()
                for coord in coords_to_tune:

                    # combination that goes straight into OLS
                    combo_OLS_dict = {self.hyperparameters[i]: [
                        coord[i]] for i in range(len(self.hyperparameters))}

                    # decide whether to search (criteria: has it been searched before)
                    if tuple(coord) in self.checked_dict:
                        self._check_already_trained_best_score(tuple(coord))
                        combo_OLS_dict['score'] = self.checked_dict[tuple(
                            coord)]['score']

                    else:
                        combo_dict = dict()  # combination that gets transformed for searching
                        for i in range(len(self.hyperparameters)):

                            # transform
                            if self.transform[self.hyperparameters[i]] == '10^':
                                combo_dict[self.hyperparameters[i]] = [
                                    10**coord[i]]

                            else:
                                combo_dict[self.hyperparameters[i]] = [
                                    coord[i]]

                        # search it
                        self._train_and_test_combo(combo_dict)
                        combo_OLS_dict['score'] = self.val_score

                        if self._is_new_best:
                            self._protective_bounds = self._get_protective_bounds(
                                old_bounds_list[k])

                        # store its metadata into checked_dict
                        self.checked_dict[tuple(coord)] = {
                            'score': self.val_score}
                        self._save_checked_dict_as_pickle()

                    # put this coord into df containing all boundaries (for later sliming depending on centre, and then OLS)
                    tmp_boundary = pd.DataFrame(combo_OLS_dict)
                    boundaries = pd.concat([boundaries, tmp_boundary])

                if len(boundaries) == 1:  # if only one coordinate in this boundary (i.e. all categorical)
                    continue

                # get the components that make up the centre (as well as new categories); and then unpack them into centres
                centre_components, new_cat = self._get_centre_components(
                    old_bounds_list[k], self.categorical)

                centres = self._unpack_centre(centre_components)

                # get the categorical features' values into a list for use in OLS preparation
                categorical_value_list = self._get_categorical(
                    new_cat, boundaries)

                for i in range(len(centres)):  # run through each different centre

                    # create a dataframe version of centre (so we could put it into OLS)
                    centre_OLS_df = pd.DataFrame(
                        {self.hyperparameters[j]: [centres[i][j]] for j in range(len(centres[i]))})

                    # decide whether to search (criteria: has it been searched before)
                    if tuple(centres[i]) in self.checked_dict:
                        self._check_already_trained_best_score(
                            tuple(centres[i]))
                        actual_centre_score = self.checked_dict[tuple(
                            centres[i])]['score']

                    else:
                        centre_df = dict()
                        for j in range(len(self.hyperparameters)):

                            # transform
                            if self.transform[self.hyperparameters[j]] == '10^':
                                centre_df[self.hyperparameters[j]] = [
                                    10**centres[i][j]]

                            else:
                                centre_df[self.hyperparameters[j]] = [
                                    centres[i][j]]

                        # search it
                        self._train_and_test_combo(centre_df)
                        actual_centre_score = self.val_score

                        # store its metadata into checked_list
                        self.checked_dict[tuple(centres[i])] = {
                            'score': self.val_score}
                        self._save_checked_dict_as_pickle()

                    # copy the boundary dataframes - to turn into the correct training data for OLS (one lm model for each centre)
                    tmp_boundary = copy.deepcopy(boundaries)
                    tmp_boundary_drop = copy.deepcopy(boundaries)

                    n_cat = 0
                    for j in range(len(new_cat)):

                        if new_cat[self.hyperparameters[j]] == True:

                            # deal with newly become categorical features

                            tmp_boundary = tmp_boundary[tmp_boundary[self.hyperparameters[j]]
                                                        == categorical_value_list[i][n_cat]]
                            tmp_boundary_drop = tmp_boundary_drop[tmp_boundary_drop[self.hyperparameters[j]]
                                                                  == categorical_value_list[i][n_cat]]
                            tmp_boundary_drop = tmp_boundary_drop.drop(
                                [self.hyperparameters[j]], axis=1)
                            centre_OLS_df = centre_OLS_df.drop(
                                [self.hyperparameters[j]], axis=1)
                            n_cat += 1

                    tmp_boundary_X = tmp_boundary_drop.drop(['score'], axis=1)
                    tmp_boundary_y = tmp_boundary_drop['score']

                    OLS = sm.OLS(tmp_boundary_y, tmp_boundary_X).fit()
                    pred_centre_score = OLS.predict(centre_OLS_df)[0]
                    print('Pred centre score:', pred_centre_score)
                    print('Actual centre score:', actual_centre_score, '\n')

                    if self._is_new_best:
                        self._protective_bounds = self._get_protective_bounds2(
                            tmp_boundary, new_cat)

                    if self._round >= 3:
                        if actual_centre_score < 0:
                            print('ACTUAL NEG AFTER 3 ROUNDS!\n')
                            continue

                    if actual_centre_score >= pred_centre_score-0.005 and actual_centre_score <= pred_centre_score+0.005:
                        print('FIT!\n')

                    else:
                        bounds_original_format = self._rebuild_bounds_to_original_format(
                            tmp_boundary, new_cat)
                        new_bounds_list = self._get_new_bounds(
                            bounds_original_format, centres[i], new_cat)

                        if new_bounds_list != False:
                            for new_bounds in new_bounds_list:
                                tmp_bounds_list.append(
                                    (new_bounds, actual_centre_score))

            tmp_bounds_list.sort(key=lambda x: x[1], reverse=True)
            n_accept = max(64, 2**len(self.hyperparameters))
            tmp_bounds_list = tmp_bounds_list[:n_accept]
            bounds_list = [x[0] for x in tmp_bounds_list]

            self._round += 1

        # CRUISE ALGORITHM
        print("Begin Cruise")

        cruise_bounds = [self._protective_bounds_to_original_bounds()]

        run_through = True

        while cruise_bounds:  # get reset every iteration, so algo will keep running if there are bounds to operate on
            print('Cruise Round')
            if run_through == True:
                old_max_bounds = cruise_bounds[0]

            old_cruise_bounds = copy.deepcopy(cruise_bounds)
            tmp_bounds_list = list()

            for k in range(len(old_cruise_bounds)):  # now run algorithm on every bound
                # get the coordinates that define the bounds
                coords_to_tune = self._get_coords_from_bounds(
                    old_cruise_bounds[k])

                # get all coordinates into a DataFrame - used for getting boundary
                boundaries = pd.DataFrame()
                for coord in coords_to_tune:

                    # combination that goes straight into OLS
                    combo_OLS_dict = {self.hyperparameters[i]: [
                        coord[i]] for i in range(len(self.hyperparameters))}

                    # decide whether to search (criteria: has it been searched before)
                    if tuple(coord) in self.checked_dict:
                        self._check_already_trained_best_score(coord)
                        combo_OLS_dict['score'] = self.checked_dict[tuple(
                            coord)]['score']

                    else:
                        combo_dict = dict()  # combination that gets transformed for searching
                        for i in range(len(self.hyperparameters)):

                            # transform
                            if self.transform[self.hyperparameters[i]] == '10^':
                                combo_dict[self.hyperparameters[i]] = [
                                    10**coord[i]]

                            else:
                                combo_dict[self.hyperparameters[i]] = [
                                    coord[i]]

                        # search it
                        self._train_and_test_combo(combo_dict)
                        combo_OLS_dict['score'] = self.val_score

                        if self._is_new_best:
                            self._protective_bounds = self._get_protective_bounds(
                                old_cruise_bounds[k])

                        # store its metadata into checked_dict
                        self.checked_dict[tuple(coord)] = {
                            'score': self.val_score}
                        self._save_checked_dict_as_pickle()

                    # put this coord into df containing all boundaries (for later sliming depending on centre, and then OLS)
                    tmp_boundary = pd.DataFrame(combo_OLS_dict)
                    boundaries = pd.concat([boundaries, tmp_boundary])

                if len(boundaries) == 1:  # if only one coordinate in this boundary (i.e. all categorical)
                    continue

                # get the components that make up the centre (as well as new categories); and then unpack them into centres
                centre_components, new_cat = self._get_centre_components(
                    old_cruise_bounds[k], self.categorical)  # 加进去 - 改 for bound bounds with index

                centres = self._unpack_centre(centre_components)

                # get the categorical features' values into a list for use in OLS preparation
                categorical_value_list = self._get_categorical(
                    new_cat, boundaries)

                for i in range(len(centres)):  # run through each different centre

                    # create a dataframe version of centre (so we could put it into OLS)
                    centre_OLS_df = pd.DataFrame(
                        {self.hyperparameters[j]: [centres[i][j]] for j in range(len(centres[i]))})

                    # decide whether to search (criteria: has it been searched before)
                    if tuple(centres[i]) in self.checked_dict:
                        self._check_already_trained_best_score(
                            tuple(centres[i]))

                    else:
                        centre_df = dict()
                        for j in range(len(self.hyperparameters)):

                            # transform
                            if self.transform[self.hyperparameters[j]] == '10^':
                                centre_df[self.hyperparameters[j]] = [
                                    10**centres[i][j]]

                            else:
                                centre_df[self.hyperparameters[j]] = [
                                    centres[i][j]]

                        # search it
                        self._train_and_test_combo(centre_df)

                        # store its metadata into checked_list
                        self.checked_dict[tuple(centres[i])] = {
                            'score': self.val_score}
                        self._save_checked_dict_as_pickle()

                    if self._is_new_best:
                        self._protective_bounds = self._get_protective_bounds2(
                            tmp_boundary, new_cat)

                    if run_through == True:
                        # run straight through - one more round
                        bounds_original_format = self._rebuild_bounds_to_original_format(
                            tmp_boundary, new_cat)
                        new_bounds_list = self._get_new_bounds(
                            bounds_original_format, centres[i], new_cat)

                        if new_bounds_list != False:
                            for new_bounds in new_bounds_list:
                                tmp_bounds_list.append(
                                    (new_bounds, actual_centre_score))

            if run_through == False:
                max_bounds = self._protective_bounds_to_original_bounds()
                if max_bounds != old_max_bounds:
                    cruise_bounds = [max_bounds]
                else:
                    cruise_bounds = []
            else:
                tmp_bounds_list.sort(key=lambda x: x[1], reverse=True)
                n_accept = max(64, 2**len(self.hyperparameters))
                tmp_bounds_list = tmp_bounds_list[:n_accept]
                cruise_bounds = [x[0] for x in tmp_bounds_list]

            run_through = not run_through

        # Display final information
        print("TUNING FINISHED\n")

        self.view_best_combo_and_score()

    def _eval_combo(self, df_building_dict, train_pred, val_pred, test_pred):

        metrics_dict = dd(float)

        if self.clf_type == 'Regression':

            try:
                metrics_dict['train_r2'] = r2_score(self.train_y, train_pred)
            except:
                pass
            try:
                metrics_dict['val_r2'] = r2_score(self.val_y, val_pred)
            except:
                pass
            try:
                metrics_dict['test_r2'] = r2_score(self.test_y, test_pred)
            except:
                pass

            try:
                metrics_dict['train_rmse'] = np.sqrt(
                    mean_squared_error(self.train_y, train_pred))
            except:
                pass
            try:
                metrics_dict['val_rmse'] = np.sqrt(
                    mean_squared_error(self.val_y, val_pred))
            except:
                pass
            try:
                metrics_dict['test_rmse'] = np.sqrt(
                    mean_squared_error(self.test_y, test_pred))
            except:
                pass

            if self.key_stats_only == True:
                try:
                    metrics_dict['train_mape'] = mean_absolute_percentage_error(
                        self.train_y, train_pred)
                except:
                    pass
                try:
                    metrics_dict['val_mape'] = mean_absolute_percentage_error(
                        self.val_y, val_pred)
                except:
                    pass
                try:
                    metrics_dict['test_mape'] = mean_absolute_percentage_error(
                        self.test_y, test_pred)
                except:
                    pass

            df_building_dict['Train r2'] = [
                np.round(metrics_dict.get('train_r2', 0), 6)]
            df_building_dict['Val r2'] = [
                np.round(metrics_dict.get('val_r2', 0), 6)]
            df_building_dict['Test r2'] = [
                np.round(metrics_dict.get('test_r2', 0), 6)]
            df_building_dict['Train rmse'] = [
                np.round(metrics_dict.get('train_rmse', 0), 6)]
            df_building_dict['Val rmse'] = [
                np.round(metrics_dict.get('val_rmse', 0), 6)]
            df_building_dict['Test rmse'] = [
                np.round(metrics_dict.get('test_rmse', 0), 6)]

            if self.key_stats_only == True:
                df_building_dict['Train mape'] = [
                    np.round(metrics_dict.get('train_mape', 0), 6)]
                df_building_dict['Val mape'] = [
                    np.round(metrics_dict.get('val_mape', 0), 6)]
                df_building_dict['Test mape'] = [
                    np.round(metrics_dict.get('test_mape', 0), 6)]

        elif self.clf_type == 'Classification':

            try:
                metrics_dict['train_accuracy'] = accuracy_score(
                    self.train_y, train_pred)
            except:
                pass
            try:
                metrics_dict['val_accuracy'] = accuracy_score(
                    self.val_y, val_pred)
            except:
                pass
            try:
                metrics_dict['test_accuracy'] = accuracy_score(
                    self.test_y, test_pred)
            except:
                pass

            try:
                metrics_dict['train_f1'] = f1_score(
                    self.train_y, train_pred, average='weighted')
            except:
                pass
            try:
                metrics_dict['val_f1'] = f1_score(
                    self.val_y, val_pred, average='weighted')
            except:
                pass
            try:
                metrics_dict['test_f1'] = f1_score(
                    self.test_y, test_pred, average='weighted')
            except:
                pass

            try:
                metrics_dict['train_precision'] = precision_score(
                    self.train_y, train_pred, average='weighted')
            except:
                pass
            try:
                metrics_dict['val_precision'] = precision_score(
                    self.val_y, val_pred, average='weighted')
            except:
                pass
            try:
                metrics_dict['test_precision'] = precision_score(
                    self.test_y, test_pred, average='weighted')
            except:
                pass

            try:
                metrics_dict['train_recall'] = recall_score(
                    self.train_y, train_pred, average='weighted')
            except:
                pass
            try:
                metrics_dict['val_recall'] = recall_score(
                    self.val_y, val_pred, average='weighted')
            except:
                pass
            try:
                metrics_dict['test_recall'] = recall_score(
                    self.test_y, test_pred, average='weighted')
            except:
                pass

            if self.key_stats_only == True:
                try:
                    metrics_dict['train_bal_accu'] = balanced_accuracy_score(
                        self.train_y, train_pred)
                except:
                    pass
                try:
                    metrics_dict['val_bal_accu'] = balanced_accuracy_score(
                        self.val_y, val_pred)
                except:
                    pass
                try:
                    metrics_dict['test_bal_accu'] = balanced_accuracy_score(
                        self.test_y, test_pred)
                except:
                    pass

                try:
                    metrics_dict['train_ap'] = average_precision_score(
                        self.train_y, train_pred)
                except:
                    pass
                try:
                    metrics_dict['val_ap'] = average_precision_score(
                        self.val_y, val_pred)
                except:
                    pass
                try:
                    metrics_dict['test_ap'] = average_precision_score(
                        self.test_y, test_pred)
                except:
                    pass

                try:
                    metrics_dict['train_auc'] = roc_auc_score(
                        self.train_y, train_pred)
                except:
                    pass
                try:
                    metrics_dict['val_auc'] = roc_auc_score(
                        self.val_y, val_pred)
                except:
                    pass
                try:
                    metrics_dict['test_auc'] = roc_auc_score(
                        self.test_y, test_pred)
                except:
                    pass

            df_building_dict['Train accuracy'] = [
                np.round(metrics_dict.get('train_accuracy', 0), 6)]
            df_building_dict['Val accuracy'] = [
                np.round(metrics_dict.get('val_accuracy', 0), 6)]
            df_building_dict['Test accuracy'] = [
                np.round(metrics_dict.get('val_accuracy', 0), 6)]
            df_building_dict['Train f1'] = [
                np.round(metrics_dict.get('train_f1', 0), 6)]
            df_building_dict['Val f1'] = [
                np.round(metrics_dict.get('val_f1', 0), 6)]
            df_building_dict['Test f1'] = [
                np.round(metrics_dict.get('test_f1', 0), 6)]
            df_building_dict['Train precision'] = [
                np.round(metrics_dict.get('train_precision', 0), 6)]
            df_building_dict['Val precision'] = [
                np.round(metrics_dict.get('val_precision', 0), 6)]
            df_building_dict['Test precision'] = [
                np.round(metrics_dict.get('test_precision', 0), 6)]
            df_building_dict['Train recall'] = [
                np.round(metrics_dict.get('train_recall', 0), 6)]
            df_building_dict['Val recall'] = [
                np.round(metrics_dict.get('val_recall', 0), 6)]
            df_building_dict['Test recall'] = [
                np.round(metrics_dict.get('test_recall', 0), 6)]

            if self.key_stats_only == True:
                df_building_dict['Train balanced_accuracy'] = [
                    np.round(metrics_dict.get('train_bal_accu', 0), 6)]
                df_building_dict['Val balanced_accuracy'] = [
                    np.round(metrics_dict.get('val_bal_accu', 0), 6)]
                df_building_dict['Test balanced_accuracy'] = [
                    np.round(metrics_dict.get('test_bal_accu', 0), 6)]
                df_building_dict['Train AP'] = [
                    np.round(metrics_dict.get('train_ap', 0), 6)]
                df_building_dict['Val AP'] = [
                    np.round(metrics_dict.get('val_ap', 0), 6)]
                df_building_dict['Test AP'] = [
                    np.round(metrics_dict.get('test_ap', 0), 6)]
                df_building_dict['Train AUC'] = [
                    np.round(metrics_dict.get('train_auc', 0), 6)]
                df_building_dict['Val AUC'] = [
                    np.round(metrics_dict.get('val_auc', 0), 6)]
                df_building_dict['Test AUC'] = [
                    np.round(metrics_dict.get('test_auc', 0), 6)]

        return df_building_dict, metrics_dict[f'val_{self.optimised_metric}'], metrics_dict[f'test_{self.optimised_metric}']

    def _train_and_test_combo(self, combo):
        """ Helper to train and test each combination as part of tune() """

        params = {self.hyperparameters[i]: combo[self.hyperparameters[i]][0] for i in range(
            len(self.hyperparameters))}

        if self._tune_features == True:
            del params['features']
            tmp_train_x = self.train_x[list(
                self._feature_combo_n_index_map[combo['features'][0]])]
            tmp_val_x = self.val_x[list(
                self._feature_combo_n_index_map[combo['features'][0]])]
            tmp_test_x = self.test_x[list(
                self._feature_combo_n_index_map[combo['features'][0]])]

            if self.pytorch_model:
                params['input_dim'] = len(
                    list(self._feature_combo_n_index_map[combo[-1]]))

            # add non tuneable parameters
            for nthp in self.non_tuneable_parameter_choices:
                params[nthp] = self.non_tuneable_parameter_choices[nthp]

            # initialise object
            clf = self.model(**params)

            params['features'] = [
                list(self._feature_combo_n_index_map[combo['features'][0]])]
            params['n_columns'] = len(
                list(self._feature_combo_n_index_map[combo['features'][0]]))
            params['n_features'] = combo['features'][0]
            params['feature combo ningxiang score'] = self.feature_n_ningxiang_score_dict[self._feature_combo_n_index_map[combo['features'][0]]]

        else:
            tmp_train_x = self.train_x
            tmp_val_x = self.val_x
            tmp_test_x = self.test_x

            if self.pytorch_model:
                params['input_dim'] = len(list(self.train_x.columns))

            # add non tuneable parameters
            for nthp in self.non_tuneable_parameter_choices:
                params[nthp] = self.non_tuneable_parameter_choices[nthp]

            # initialise object
            clf = self.model(**params)

        # get time and fit
        start = time.time()
        clf.fit(tmp_train_x, self.train_y)
        end = time.time()

        # get predicted labels/values for three datasets
        train_pred = clf.predict(tmp_train_x)
        val_pred = clf.predict(tmp_val_x)
        test_pred = clf.predict(tmp_test_x)

        # get scores and time used
        time_used = end-start

        # build output dictionary and save result
        df_building_dict = params

        # get evaluation statistics
        df_building_dict, val_score, test_score = self._eval_combo(
            df_building_dict, train_pred, val_pred, test_pred)

        df_building_dict['Time'] = [np.round(time_used, 2)]
        df_building_dict['Precedence'] = [self._up_to]

        tmp = pd.DataFrame(df_building_dict)

        self.tuning_result = pd.concat([self.tuning_result, tmp])
        self.tuning_result.index = range(len(self.tuning_result))
        self._save_tuning_result()

        self._is_new_best = 0

        # update best score stats
        if val_score > self.best_score:
            self.best_score = val_score
            self.best_clf = clf
            self.best_combo = combo

            if self.best_model_saving_address:
                self._save_best_model()

            self._is_new_best = 1

        # add a new self variable compared to previous JiaXing classes
        self.val_score = val_score

        self._up_to += 1

        print(f'''Trained and Tested combination {self._up_to}: {combo}, taking {np.round(time_used, 2)} seconds to get val score of {np.round(val_score, 4)}
        Current best combo: {self.best_combo} with val score {np.round(self.best_score, 4)}''')

    def _save_tuning_result(self):
        """ Helper to export tuning result csv """

        # check twice to strip everything - they share one address
        tuning_result_saving_address_strip = self.tuning_result_saving_address.split('.csv')[
            0]
        tuning_result_saving_address_strip = tuning_result_saving_address_strip.split('.pickle')[
            0]

        self.tuning_result.to_csv(
            f'{tuning_result_saving_address_strip}.csv', index=False)

    def _save_checked_dict_as_pickle(self):
        """ Helper to export checked dict as pickle """

        # check twice to strip everything - they share one address
        tuning_result_saving_address_strip = self.tuning_result_saving_address.split('.csv')[
            0]
        tuning_result_saving_address_strip = tuning_result_saving_address_strip.split('.pickle')[
            0]

        with open(f'{tuning_result_saving_address_strip}.pickle', 'wb') as f:
            pickle.dump(self.checked_dict, f)

    def view_best_combo_and_score(self):
        """ View best combination and its validation score """

        print('Max Val Score: \n', self.best_score)

        max_val_id = self.tuning_result[f'Val {self.optimised_metric}'].idxmax(
        )
        print('Best Combo Test Score: \n',
              self.tuning_result.iloc[max_val_id][f'Test {self.optimised_metric}'])
        print('Best Combo Train Score: \n',
              self.tuning_result.iloc[max_val_id][f'Train {self.optimised_metric}'])

        print('Max Combo Index: \n', self.best_combo)

        final_combo = {self.hyperparameters[i]: self.best_combo[i] for i in range(
            len(self.hyperparameters))}
        print('Max Combo Hyperparamer Combination: \n', final_combo)

        if self._tune_features:
            print('Max Combo Features: \n',
                  self._feature_combo_n_index_map[self.best_combo[-1]])

        print('# Combos Checked:', int(len(self.checked_dict)))

    def read_in_tuning_result_df(self, df_address, object_address):
        """ Read in checked dict from outputted pickle object """

        self.tuning_result = pd.read_csv(df_address)

        with open(object_address, 'rb') as f:
            self.checked_dict = pickle.load(f)

        self._up_to = len(self.checked_dict)

        print(
            f"Successfully read in tuning result of {len(self.checked_dict)} rows")

    def _check_already_trained_best_score(self, combo):
        """ Helper for checking whether an already trained combo is best score """

        combo = tuple(combo)

        self.is_new_best = 0

        # update best score stats
        if self.checked_dict[combo]['score'] > self.best_score:
            self.best_score = self.checked_dict[combo]['score']
            self.best_clf = None
            print(
                f"As new Best Combo {combo} was read in, best_clf is set to None")
            self.best_combo = combo

            self._is_new_best = 1

        print(f'''Already Trained and Tested combination { {self.hyperparameters[i]:combo[i] for i in range(len(combo))} }, which had val score of {np.round(self.checked_dict[combo]['score'], 4)}
        Current best combo: {self.best_combo} with val score {np.round(self.best_score, 4)}. 
        Has trained {self._up_to} combinations so far''')

    def _str_to_list(self, string):
        """ Helper to convert string to list"""

        out = list()
        for feature in string.split(', '):
            out.append(feature.strip('[').strip(']').strip("'"))

        return out

    def set_tuning_result_saving_address(self, address):
        """ Read in where to save tuning result dataframe """

        self.tuning_result_saving_address = address
        print('Successfully set tuning output address')

    def set_best_model_saving_address(self, address):
        """ Read in where to save best model """

        self.best_model_saving_address = address
        print('Successfully set best model output address')

    def _save_best_model(self):
        """ Helper to save best model as a pickle """

        best_model_saving_address_split = self.best_model_saving_address.split('.pickle')[
            0]

        with open(f'{best_model_saving_address_split}.pickle', 'wb') as f:
            pickle.dump(self.best_clf, f)
