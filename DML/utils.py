# from DOWHY
import ast

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor


def get_numeric_features(X):
    """
    Finds the numeric feature columns in a dataset

    :param X: pandas dataframe

    returns: list of indices of numeric features
    """
    numeric_features_names = list(X.select_dtypes("number"))
    numeric_features = []
    for col_name in numeric_features_names:
        col_index = X.columns.get_loc(col_name)
        numeric_features.append(col_index)
    return numeric_features


def _parse_params(params):
    """
    params_t / params_y arrive from the CSV as string reprs of dicts
    (see model_robustness master_robustness_results.csv). Accept
    either a dict already, a string repr of one, or None.
    """
    if params is None:
        return None
    if isinstance(params, str):
        return ast.literal_eval(params)
    return params


def select_model(algo, params=None):
    """
    Instantiates the *already-selected* nuisance model (chosen upstream by
    model_robustness / causality_analysis) rather than searching over a
    candidate list. Kept identical in spirit to causality.py's select_model
    so the exact same algorithm/hyperparameters used to produce t_resid /
    agg_resid are the ones reused for benchmarking-covariate refits.

    :param algo: one of 'lgbm', 'rf', 'xgb'
    :param params: dict (or string repr of dict) of hyperparameters, or None
    """
    params = _parse_params(params) or {}
    if algo == "lgbm":
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            **params,
        )
    if algo == "rf":
        return RandomForestRegressor(
            n_estimators=500, random_state=42, n_jobs=-1, **params
        )
    if algo == "xgb":
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            **params,
        )
    raise ValueError(f"Unknown algo '{algo}'. Expected one of 'lgbm', 'rf', 'xgb'.")


def get_regression_r2(X, Y, split_indices, algo, params):

    num_samples = X.shape[0]
    pred = np.zeros(num_samples)
    for train, test in split_indices:
        model = select_model(algo, params)
        model.fit(X[train], Y[train].ravel())
        pred[test] = model.predict(X[test])
    return np.var(pred) / np.var(Y)
