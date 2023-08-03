import json
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


import features as FEAT
from metrics import absolute_error, squared_error
from mnh import MNH_load, df_mnh_vf
from voice_features import vf_load, stand_vf

def SLR_sklearn(X, y, f, sex, age_min, age_max, R_2_min):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    ## Transform data
    # poly = PolynomialFeatures(degree=max_poly_degr, include_bias=False)
    # poly.fit(X_train)
    # X_train_tf = poly.transform(X_train)
    # X_train = X_train_tf
    # X_test_tf = poly.transform(X_test)
    # X_test = X_test_tf

    lin_reg = LinearRegression()  # create an instance of the LinearRegression class
    lin_reg.fit(X_train, y_train)  # fit training data

    # coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
    # print(coeff_df)
    # print('Intercept: \n', lin_reg.intercept_)
    # print('Coefficients: \n', lin_reg.coef_)
    y_pred_train = lin_reg.predict(X_train)
    y_pred_test = lin_reg.predict(X_test)

    # df_result_train = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred_train})
    # print("Train",df_result_train)
    # df_result_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
    # print("Test", df_result_test)

    ## Compute the MSE for the training set and the test set (i.e. the training error and the test error)
    MAE_train = mean_absolute_error(y_train, y_pred_train)
    MAE_test = mean_absolute_error(y_test, y_pred_test)

    RMSE_train = r2_score(y_train, y_pred_train)
    RMSE_test = r2_score(y_test, y_pred_test)

    MSE_train = mean_squared_error(y_train, y_pred_train)
    MSE_test = mean_squared_error(y_test, y_pred_test)

    def err(y_true, y_pred):
        oe = np.std((y_true - y_pred) ** 2, axis=0)
        return oe

    print("Linear Regression")
    print("MSE training set:", MSE_train, err(y_train, y_pred_train), len(X_train))
    print("MSE test set:", MSE_test, err(y_test, y_pred_test), len(X_test))



def load_data(data_dir):
    df_vf, pat = vf_load(data_dir) ## voice features
    path_demo_csv = os.path.join(data_dir, "PID_info.csv")
    df_demo = pd.read_csv(path_demo_csv) ## patient demographics
    df_MNH, mnh_X = MNH_load(data_dir) ## MNH data

    return df_vf, pat, df_demo, df_MNH


def load_group(df_vf, df_demo, sex, age_min, age_max):
    if sex in [1, 2]:
        df_gender = df_demo[(df_demo['sex'] == sex)]
    else:
        df_gender = df_demo

    df_age = df_gender[(df_demo['age'] >= age_min) & (df_demo['age'] <= age_max)]
    df_group = df_vf[df_vf.index.isin(df_age['PID'])]
    #stand_vf(df_group, df_group.columns[13:])
    df_group = df_group.reset_index()
    df_group.rename(columns={"index": "PID"}, inplace=True)
    return df_group


def calculate_scores(y_pred, y_true):
    ## Compute the MSE for the training set and the test set (i.e. the training error and the test error)
    return {
        "ae": absolute_error(y_pred, y_true).squeeze(),
        "mse": squared_error(y_pred, y_true).squeeze(),
        "r2": r2_score(y_true, y_pred, multioutput='raw_values')
    }


def scores_summary(scores_dict, title=""):
    summary = []
    summary.append(f"==== Scores for {title}")
    for score_name, scores in scores_dict.items():
        summary.append(f"{score_name}:  {np.mean(scores):.4f} (std: {np.std(scores):.4f}, N={len(scores)})")
    return summary


def run_model(X, y, model_cls=RandomForestRegressor, model_kwargs={}, normalize_x=True, normalize_y=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    if normalize_y:
        y_min, y_max = y_train.min(), y_train.max()
        y_range = y_max - y_min
        y_train = (y_train - y_min) / y_range
        y_test = (y_test - y_min) / y_range

    model = model_cls(**model_kwargs)  # create an instance of the LinearRegression class

    if normalize_x:
        pipe = make_pipeline(preprocessing.StandardScaler(), model)
    else:
        pipe = model
    pipe.fit(X_train, y_train)  # fit training data

    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    scores_dict_train = calculate_scores(y_pred=y_pred_train, y_true=y_train)
    scores_dict_test = calculate_scores(y_pred=y_pred_test, y_true=y_test)
    return scores_dict_train, scores_dict_test


def calculate_scores(y_pred, y_true):
    ## Compute the MSE for the training set and the test set (i.e. the training error and the test error)
    return {
        "ae": absolute_error(y_pred, y_true).squeeze(),
        "mse": squared_error(y_pred, y_true).squeeze(),
        "r2": r2_score(y_pred, y_true, multioutput='raw_values')
    }


def scores_summary(scores_dict, title=""):
    summary = []
    summary.append(f"==== Scores for {title}")
    for score_name, scores in scores_dict.items():
        summary.append(f"{score_name}:  {np.mean(scores):.4f} (std: {np.std(scores):.4f}, N={len(scores)})")
    return summary


def save_scores_dict(scores_dict, folder, filename):
    scores_dict_filtered = {k: v for k, v in scores_dict.items() if len(v) > 1}
    pd.DataFrame(scores_dict_filtered).to_csv(os.path.join(folder, filename))


def save_results(scores_dict_train, scores_dict_test, config, verbose=False):
    experiment_dir = os.path.join(config["out_dir"], config["experiment_name"])
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        for k, v in config.items():
            if hasattr(v, 'fit'):
                config[k] = str(v.__name__)
        json.dump(config, f)

    save_scores_dict(scores_dict_train, experiment_dir, "train.csv")
    save_scores_dict(scores_dict_test, experiment_dir, "test.csv")

    summary = [f"==== Summary for {config['experiment_name']}"]
    summary += scores_summary(scores_dict_train, "train")
    summary += scores_summary(scores_dict_test, "test")
    summary_str = f"{os.linesep}".join(summary)
    with open(os.path.join(experiment_dir, "summary.txt"), "w") as f:
        f.write(summary_str)
    if verbose:
        print(summary_str)


def run(config):
    df_vf, pat, df_demo, df_MNH = load_data(config["data_dir"])
    # "row selection", i.e., dataset filtering
    df_group = load_group(df_vf=df_vf, df_demo=df_demo, **config["group"])
    # Combine dfs
    mv = df_mnh_vf(df_MNH, df_vf, df_group)
    mv = mv.dropna(subset=['mnh'])
    mv.set_index('PID', inplace=True)

    X = mv[config["features"]]
    y = mv.values[:, 3]

    scores_dict_train, scores_dict_test = run_model(X, y, model_cls=config["model_cls"],
                                                    model_kwargs=config["model_kwargs"],
                                                    normalize_x=config["normalize_x"],
                                                    normalize_y=config["normalize_y"])
    save_results(scores_dict_train, scores_dict_test, config, verbose=config["verbose"])


class NaiveRegressor(LinearRegression):
    def fit(self, X, y, sample_weight=None):
        self.mean = np.mean(y)

    def predict(self, X):
        return np.ones(X.shape[0]) * self.mean


def run_all():
    # Configs is a list of tuples: (model class, model_kwargs, experiment name)
    configs = [
        (DecisionTreeRegressor, {}, "DT"),
        (RandomForestRegressor, {"n_estimators": 100}, "RF"),
        (SVR, {}, "SVM"),
        (GaussianMixture, {"n_components": 5}, "GMM"),
        (GaussianProcessRegressor, {}, "GPR_default"),
        (MLPRegressor, {"hidden_layer_sizes": (10,)}, "MLP_10"),  # Should use normalized / scaled target vars y
        (MLPRegressor, {"hidden_layer_sizes": (10, 10)}, "MLP_10_10"),
        (MLPRegressor, {"hidden_layer_sizes": (10, 10, 10)}, "MLP_10_10_10"),
        (MLPRegressor, {"hidden_layer_sizes": (20,)}, "MLP_20"),
    ]
    for model_cls, model_kwargs, experiment_name in configs:
        config = {
            "data_dir": os.path.expanduser(os.path.join("~", "Documents", "ETH MA Data", "play", "CAMP_study_data")),
            "out_dir": os.path.expanduser(os.path.join("~", "Documents", "ETH MA Data", "play", "ML Results")),
            "experiment_name": experiment_name,
            "group": {
                "sex": 'all',
                "age_min": -1,
                "age_max": 999
            },
            "features": [FEAT.f0, FEAT.ji, FEAT.shi, FEAT.mfcc1],
            "model_cls": model_cls,
            "model_kwargs": model_kwargs,
            "normalize_x": True,
            "normalize_y": True,
            "verbose": False
        }
        run(config)


if __name__ == "__main__":
    # Run a single config
    config = {
        "data_dir": os.path.expanduser(os.path.join("~", "Documents", "ETH MA Data", "play", "CAMP_study_data")),
        "out_dir": os.path.expanduser(os.path.join("~", "Documents", "ETH MA Data", "play", "ML Results")),
        "experiment_name": "randomforest_allrows",
        "group": {
            "sex": 'all',
            "age_min": -1,
            "age_max": 999
        },
        "features": [FEAT.f0, FEAT.ji, FEAT.shi, FEAT.mfcc1],
        # "model_cls": RandomForestRegressor,
        "model_cls": NaiveRegressor,
        "model_kwargs": {},
        "normalize_x": True,
        "normalize_y": True,
        "verbose": True
    }
    run(config)

    # Run a set of configs
    #run_all()
