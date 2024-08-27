def smoke_test():

    try:
        print("===Begin Smoke Test===\n")

        print("===Import Modules===")
        import JXAutoML

        print("===Import Successful===")

        print("===Import NingXiang===")

        from JXAutoML.NingXiang import NingXiang

        print("===Import NingXiang Successful===")

        print("===Import JiaoCheng===")

        from JXAutoML.JiaoCheng import JiaoCheng

        print("===Import JiaoCheng Successful===")

        print("===Import JiaoChengB===")

        from JXAutoML.JiaoChengB import JiaoChengB

        print("===Import JiaoChengB Successful===")

        print("===Import YangZhou===")

        from JXAutoML.YangZhouB import YangZhouB

        print("===Import YangZhou Successful===")

        print("======\n")
        print("===SMOKETEST PASSED===")

    except Exception as e:
        print("===SMOKETEST FAILED===")
        raise e


def test_NingXiang(X, y, mode):
    from JXAutoML.NingXiang import NingXiang

    feature_selector = NingXiang()
    feature_selector.read_in_train_data(X, y)
    feature_selector.set_model_type(mode)

    feature_order_dict = feature_selector.get_rf_based_feature_combinations(n_jobs=-1)

    feature_selector.show_rf_stats()

    return feature_order_dict


def test_YangZhou(
    train_X,
    train_y,
    val_X,
    val_y,
    test_X,
    test_y,
    mode,
    pytorch_model,
    feature_importance_ordering,
    TUNE_FEATURES_AS_HYPERPARAMETERS,
    optimised_metric,
    **kwargs,
):
    from JXAutoML.YangZhouB import YangZhouB as tuner

    if mode == "Regression":
        from sklearn.ensemble import RandomForestRegressor as clf
    elif mode == "Classification":
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
        clf, mode, pytorch_model=pytorch_model, optimised_metric=optimised_metric
    )

    # read in the data for training and validation
    tuner.read_in_data(train_X, train_y, val_X, val_y, test_X, test_y)

    # set what hp values to tune
    tuner.set_hyperparameters(parameter_choices)
    # WARNING: this may take a while if no. tuneable hyperparameters are large

    # set up hp values that need to be changed from default but NOT to be tuned
    tuner.set_non_tuneable_hyperparameters(non_tunable_hyperparameters_dict)

    # set up feature importance ordering

    if TUNE_FEATURES_AS_HYPERPARAMETERS:
        tuner.set_features(feature_importance_ordering)
        # WARNING: this may take a while if no. tuneable hyperparameters are large

    # set up where to save the tuning result csv
    tuner.set_tuning_result_saving_address(
        f"local_test_output/yangzhou_test_tuning_result.csv"
    )

    # set up where to save the current best model
    tuner.set_best_model_saving_address(
        f"local_test_output/yangzhou_test_tuning_best_model"
    )

    tuner.tune()


def test_JiaoCheng(
    train_X,
    train_y,
    val_X,
    val_y,
    test_X,
    test_y,
    mode,
    pytorch_model,
    feature_importance_ordering,
    TUNE_FEATURES_AS_HYPERPARAMETERS,
    optimised_metric,
    JiaoChengB,
    **kwargs,
):
    if JiaoChengB:
        from JXAutoML.JiaoChengB import JiaoChengB as tuner
    else:
        from JXAutoML.JiaoCheng import JiaoCheng as tuner
    if mode == "Regression":
        from sklearn.ensemble import RandomForestRegressor as clf
    elif mode == "Classification":
        from sklearn.ensemble import RandomForestClassifier as clf

    # what values to try for each hyperparameter
    parameter_choices = {
        "max_depth": (3, 6, 12, 24),
        "max_samples": (0.4, 0.55, 0.7, 0.85),
    }

    # what values to set non-tuneable parameters/hyperparameters
    non_tunable_hyperparameters_dict = {"random_state": 42, "n_jobs": -1}

    tuning_order = (
        ["features", "max_depth", "max_samples"]
        if TUNE_FEATURES_AS_HYPERPARAMETERS
        else ["max_depth", "max_samples"]
    )
    default_hyperparameter_values = (
        {"features": 0, "max_depth": 3, "max_samples": 0.4}
        if TUNE_FEATURES_AS_HYPERPARAMETERS
        else {"max_depth": 3, "max_samples": 0.4}
    )

    tuner = tuner()

    # define what model we are tuning
    tuner.read_in_model(
        clf, mode, pytorch_model=pytorch_model, optimised_metric=optimised_metric
    )

    # read in the data for training and validation
    tuner.read_in_data(train_X, train_y, val_X, val_y, test_X, test_y)

    # set what hp values to tune
    tuner.set_hyperparameters(parameter_choices)
    # WARNING: this may take a while if no. tuneable hyperparameters are large

    # set up hp values that need to be changed from default but NOT to be tuned
    tuner.set_non_tuneable_hyperparameters(non_tunable_hyperparameters_dict)

    # set up feature importance ordering

    if TUNE_FEATURES_AS_HYPERPARAMETERS:
        tuner.set_features(feature_importance_ordering)
        # WARNING: this may take a while if no. tuneable hyperparameters are large

    # set up the order of hyperparameters when iteratively tuning using JiaoCheng
    tuner.set_tuning_order(tuning_order)

    # set up the default hp values for first iteration of tuning JiaoCheng
    tuner.set_hyperparameter_default_values(default_hyperparameter_values)

    # set up where to save the tuning result csv
    tuner.set_tuning_result_saving_address(
        f'local_test_output/jiaocheng{"b" if JiaoChengB else ""}_test_tuning_result.csv'
    )

    # set up where to save the current best model
    tuner.set_best_model_saving_address(
        f'local_test_output/jiaocheng{"b" if JiaoChengB else ""}_test_tuning_best_model'
    )

    tuner.tune()


def local_test():
    # Create a regression dataset
    from sklearn.datasets import make_regression, make_classification
    from sklearn.model_selection import train_test_split
    import pandas as pd

    X_reg, y_reg = make_regression(
        n_samples=100, n_features=5, noise=0.1, random_state=42
    )
    X_reg_df = pd.DataFrame(
        X_reg, columns=[f"feature_{i+1}" for i in range(X_reg.shape[1])]
    )
    y_reg_series = pd.Series(y_reg, name="target")

    X_reg_df_train, X_reg_df_valtest, y_reg_series_train, y_reg_series_valtest = (
        train_test_split(X_reg_df, y_reg_series, test_size=0.3, random_state=42)
    )
    X_reg_df_val, X_reg_df_test, y_reg_series_val, y_reg_series_test = train_test_split(
        X_reg_df_valtest, y_reg_series_valtest, test_size=0.5, random_state=42
    )

    # Create a classification dataset with 2 classes
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

    smoke_test()

    print("===Begin NingXiang Test===\n")
    print("===Begin NingXiang Regression Test===")
    reg_feature_importance = test_NingXiang(
        X_class_2_df_train, y_class_2_series_train, "Regression"
    )
    print("===Test NingXiang Regression Successful===")
    print("===Begin NingXiang Classification Test===")
    class_2_feature_importance = test_NingXiang(
        X_class_2_df_val, y_class_2_series_val, "Classification"
    )
    print("===Test NingXiang Classification Successful===\n")
    print("===Test NingXiang Successful===\n\n")

    print("===Begin YangZhou Test===\n")
    print("===Begin YangZhou Regression (No Features) Test===")
    YangZhou_test_json_reg = {
        "train_X": X_reg_df_train,
        "train_y": y_reg_series_train,
        "val_X": X_reg_df_val,
        "val_y": y_reg_series_val,
        "test_X": X_reg_df_test,
        "test_y": y_reg_series_test,
        "mode": "Regression",
        "pytorch_model": False,
        "feature_importance_ordering": reg_feature_importance,
        "TUNE_FEATURES_AS_HYPERPARAMETERS": False,
        "optimised_metric": "r2",
    }
    test_YangZhou(**YangZhou_test_json_reg)
    print("===Test YangZhou Regression (No Features) Successful===\n")

    print("===Begin YangZhou Classification (With Features) Test===")
    YangZhou_test_json_class_2 = {
        "train_X": X_class_2_df_train,
        "train_y": y_class_2_series_train,
        "val_X": X_class_2_df_val,
        "val_y": y_class_2_series_val,
        "test_X": X_class_2_df_test,
        "test_y": y_class_2_series_test,
        "mode": "Classification",
        "pytorch_model": False,
        "feature_importance_ordering": class_2_feature_importance,
        "TUNE_FEATURES_AS_HYPERPARAMETERS": True,
        "optimised_metric": "accuracy",
    }
    test_YangZhou(**YangZhou_test_json_class_2)
    print("===Test YangZhou Classification (With Features) Successful===\n")

    print("===Test YangZhou Successful===\n\n")

    print("===Begin JiaoCheng Test===\n")

    print("===Begin JiaoCheng Regression (No Features) Test===")
    JiaoCheng_test_json_reg = {
        "train_X": X_reg_df_train,
        "train_y": y_reg_series_train,
        "val_X": X_reg_df_val,
        "val_y": y_reg_series_val,
        "test_X": X_reg_df_test,
        "test_y": y_reg_series_test,
        "mode": "Regression",
        "pytorch_model": False,
        "feature_importance_ordering": reg_feature_importance,
        "TUNE_FEATURES_AS_HYPERPARAMETERS": False,
        "optimised_metric": "r2",
        "JiaoChengB": False,
    }
    test_JiaoCheng(**JiaoCheng_test_json_reg)
    print("===Test JiaoCheng Regression (No Features) Successful===")

    print("===Begin JiaoCheng Classification (With Features) Test===")
    JiaoCheng_test_json_class_2 = {
        "train_X": X_class_2_df_train,
        "train_y": y_class_2_series_train,
        "val_X": X_class_2_df_val,
        "val_y": y_class_2_series_val,
        "test_X": X_class_2_df_test,
        "test_y": y_class_2_series_test,
        "mode": "Classification",
        "pytorch_model": False,
        "feature_importance_ordering": class_2_feature_importance,
        "TUNE_FEATURES_AS_HYPERPARAMETERS": True,
        "optimised_metric": "accuracy",
        "JiaoChengB": False,
    }
    test_JiaoCheng(**JiaoCheng_test_json_class_2)
    print("===Test JiaoCheng Classification (With Features) Successful===\n")

    print("===Test JiaoCheng Successful===\n\n")

    print("===Begin JiaoChengB Test===\n")

    print("===Begin JiaoChengB Regression (With Features) Test===")
    JiaoChengB_test_json_reg = {
        "train_X": X_reg_df_train,
        "train_y": y_reg_series_train,
        "val_X": X_reg_df_val,
        "val_y": y_reg_series_val,
        "test_X": X_reg_df_test,
        "test_y": y_reg_series_test,
        "mode": "Regression",
        "pytorch_model": False,
        "feature_importance_ordering": reg_feature_importance,
        "TUNE_FEATURES_AS_HYPERPARAMETERS": True,
        "optimised_metric": "r2",
        "JiaoChengB": True,
    }
    test_JiaoCheng(**JiaoChengB_test_json_reg)
    print("===Test JiaoChengB Regression (With Features) Successful===")

    print("===Begin JiaoChengB Classification (No Features) Test===")
    JiaoChengB_test_json_class_2 = {
        "train_X": X_class_2_df_train,
        "train_y": y_class_2_series_train,
        "val_X": X_class_2_df_val,
        "val_y": y_class_2_series_val,
        "test_X": X_class_2_df_test,
        "test_y": y_class_2_series_test,
        "mode": "Classification",
        "pytorch_model": False,
        "feature_importance_ordering": class_2_feature_importance,
        "TUNE_FEATURES_AS_HYPERPARAMETERS": False,
        "optimised_metric": "accuracy",
        "JiaoChengB": True,
    }
    test_JiaoCheng(**JiaoChengB_test_json_class_2)
    print("===Test JiaoChengB Classification (No Features) Successful===\n")

    print("===Test JiaoChengB Successful===\n\n")

    print("===")


if __name__ == "__main__":
    local_test()
    print("ALL TESTS PASSED!")
