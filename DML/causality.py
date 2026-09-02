import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import ast
import warnings

import lightgbm as lgb  # non-linear modeling
import numpy as np
import pandas as pd
import scipy
import sensemakr as smkr
import statsmodels.formula.api as smf
import xgboost as xgb
from DML.non_parametric_sensitivity import sensitivity_analysis
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor

# For DML estimation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # bootsratp progress

##
from process_dataset import *
from visualisations import *

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Bootstrap is a resampling technique used to estimate the confidence intervals of a statistic
# (like mean or regression coefficient). It recomputes the statistic many times with random resampling
# and then builds a disitrbution of the statistics from whcih we can compute CI, varaince etc.
# Good when data is not Gaussian

# Plot how much load varies in terms of temperature

# How does load change in terms of hour of day and temperature - Predicted disitrbutional effects


# PEET temperature analysis
def assemble_feeder_data(dir, csv_directory):
    feeder = pd.read_csv(dir)

    cols_to_fix = [
        "direct_radiation_W/m2",
        "diffuse_radiation_W/m2",
        "precipitation_kg/m2",
        "snow_height_m",
        "humidity_kg/kg",
        "max_wind_gust_m/s",
        "meridional_wind_m/s",
        "zonal_wind_m/s",
    ]

    # Rename by replacing '/' with '_'
    feeder = feeder.rename(columns={col: col.replace("/", "_") for col in cols_to_fix})
    feeder = feeder.rename(columns={"Power(kW)": "Power"})
    HP_causality = feeder[
        [
            "feeder",
            "Date",
            "Power",
            "direct_radiation_W_m2",
            "diffuse_radiation_W_m2",
            "air_temperature_C",
            "precipitation_kg_m2",
            "snow_height_m",
            "humidity_kg_kg",
            "max_wind_gust_m_s",
            "meridional_wind_m_s",
            "zonal_wind_m_s",
            "Day_of_Week",
            "Month",
            "Holiday",
            "Time",
            "housing_units_count",
            "industry_and_commerce_count",
            "storage_heaters_kW",
            "heat_pumps_kW",
            "electric_heaters_kW",
            "hot_water_tanks_kW",
            "flow-type_heaters_kW",
            "other_consumers_kW",
            "batteries_kW",
            "PV_systems_kW",
        ]
    ]

    starmap_parameters = []
    feeder_id = HP_causality["feeder"].unique()
    skip_feeder = []
    for file in os.listdir(csv_directory):
        if file.startswith("uq_results_causal_temp_hp"):
            df_name = os.path.splitext(file)[0]
            df_name = df_name.split("_")[-1]
            skip_feeder.append(int(df_name))

    filter_f = [f for f in feeder_id if f not in skip_feeder]
    for f in filter_f:
        HP_causality_f = HP_causality[HP_causality["feeder"] == f].reset_index(
            drop=True
        )

        HP_causality_f["hour"] = pd.to_datetime(
            HP_causality_f["Time"], format="%H:%M:%S"
        ).dt.hour
        HP_causality_f["month"] = pd.to_datetime(
            HP_causality_f["Date"], format="%Y-%m-%d"
        ).dt.month
        HP_causality_f["day"] = pd.to_datetime(
            HP_causality_f["Date"], format="%Y-%m-%d"
        ).dt.dayofweek

        confounders = [
            "direct_radiation_W_m2",
            "diffuse_radiation_W_m2",
            "precipitation_kg_m2",
            "snow_height_m",
            "humidity_kg_kg",
            "max_wind_gust_m_s",
            "meridional_wind_m_s",
            "zonal_wind_m_s",
            "day",
            "month",
            "Holiday",
            "hour",
            "PV_systems_kW",
        ]
        treatment = "air_temperature_C"
        outcome = "Power"
        control = "heat_pumps_kW"

        # Manual CATE
        confounders_t = [
            "direct_radiation_W_m2",
            "diffuse_radiation_W_m2",
            "precipitation_kg_m2",
            "snow_height_m",
            "humidity_kg_kg",
            "max_wind_gust_m_s",
            "meridional_wind_m_s",
            "zonal_wind_m_s",
            "day",
            "month",
            "Holiday",
            "hour",
            "PV_systems_kW",
        ]
        confounders_y = [
            "direct_radiation_W_m2",
            "diffuse_radiation_W_m2",
            "precipitation_kg_m2",
            "snow_height_m",
            "humidity_kg_kg",
            "max_wind_gust_m_s",
            "meridional_wind_m_s",
            "zonal_wind_m_s",
            "day",
            "month",
            "Holiday",
            "hour",
            "PV_systems_kW",
        ]
        eq = "y_res ~ t_res"
        starmap_parameters.append(
            (
                HP_causality_f,
                f,
                confounders_t,
                confounders_y,
                treatment,
                outcome,
                control,
                eq,
            )
        )

    return starmap_parameters


def run_cross_feeder(causal_path, csv_directory):
    # Extract feeder size (residential + industry)
    all_feeder = pd.read_csv(causal_path)
    all_feeder = all_feeder[
        [
            "feeder",
            "housing_units_count",
            "industry_and_commerce_count",
            "heat_pumps_kW",
        ]
    ]

    all_results_dfs = {}
    # Get all CSV file paths in the directory
    for file in os.listdir(csv_directory):
        if file.startswith("uq_results_causal_temp_hp"):
            try:
                df_name = os.path.splitext(file)[0]
                df_name = df_name.split("_")[-1]
                all_results_dfs[df_name] = pd.read_csv(csv_directory + "\\" + file)

            except pd.errors.EmptyDataError:
                # If the file is completely empty, create a blank DataFrame
                all_results_dfs[df_name] = pd.DataFrame()
                print(f"Loaded {file} as an empty DataFrame.")

        slopes = {}
        for name, res in all_results_dfs.items():
            if not res.empty:
                # plot_CATE_with_sensitivity(res, name)
                # plot_CATE_time(res, name)
                plot_seasonal_hourly_cate(res, name)
                res = (
                    res.groupby("mean_t").mean().reset_index()
                )  # Average across windows of same mean temp to get clearer trend, then reindex for plotting

                # Extract slope
                if not (res["control"].nunique() == 1):
                    capacities = res["control"].unique()
                    for cap in sorted(capacities, reverse=True):
                        at_cap = res[res["control"] == cap].copy()
                        cold_threshold_cap = np.percentile(res["mean_t"], 10)
                        cold_windows_cap = at_cap[
                            at_cap["mean_t"] <= cold_threshold_cap
                        ]
                        robust_cold_cap = cold_windows_cap[
                            cold_windows_cap["rv"] >= 0.1
                        ]
                        if not robust_cold_cap.empty:
                            print(f"Using cold windows at capacity {cap} instead.")
                            feeder = all_feeder[
                                (all_feeder["feeder"] == int(name))
                                & (all_feeder["heat_pumps_kW"] == round(float(cap), 2))
                            ]

                            feeder_size = int(
                                feeder[["housing_units_count"]].max()
                            ) + int(feeder[["industry_and_commerce_count"]].max())
                            robust_cold = robust_cold_cap

                            break

                    worst_cate = robust_cold["cate_linear"].mean()
                    cold_p10 = at_cap["cate_linear"].min()
                    rv_mean = (
                        robust_cold["rv"].mean()
                        if "rv" in robust_cold.columns
                        else np.nan
                    )
                    print(
                        f"Feeder {name} - Feeder Size {feeder_size} -Worst-case CATE: {worst_cate}, Cold CATE p10: {cold_p10}, RV mean: {rv_mean}, Number of cold windows: {len(robust_cold)}"
                    )
                    n_cold = len(robust_cold)
                    cap_range = res["control"].max() - res["control"].min()

                    slopes[int(name)] = {
                        "values": res["cate_linear"],
                        "capacity": res["control"].unique(),
                        "effect_min": res.groupby(["control"])["cate_linear"].min(),
                        "effect_max": res.groupby(["control"])["cate_linear"].max(),
                        "max_capacity": cap,
                        "cap_range": cap_range,
                        "worst_case_cate": worst_cate,
                        "cold_cate_p10": cold_p10,
                        "rv_mean": rv_mean,
                        "n_cold_windows": n_cold,
                        "feeder_size": feeder_size,
                    }

            else:
                # Max Cap
                print("Only one capacity level, using max cap analysis.")
                cap = res["control"].unique()
                cold_threshold = np.percentile(res["mean_t"], 10)
                # windows with cold temps at max cap
                cold_windows = res[res["mean_t"] <= cold_threshold]
                robust_cold = cold_windows[cold_windows["rv"] >= 0.1]
                feeder = all_feeder[(all_feeder["feeder"] == int(name))]

                feeder_size = int(feeder[["housing_units_count"]].max()) + int(
                    feeder[["industry_and_commerce_count"]].max()
                )
                worst_cate = robust_cold["cate_linear"].mean()
                cold_p10 = at_cap["cate_linear"].min()
                rv_mean = (
                    robust_cold["rv"].mean() if "rv" in robust_cold.columns else np.nan
                )
                n_cold = len(robust_cold)
                print(
                    f"Feeder {name} -- Feeder Size {feeder_size} - Worst-case CATE: {worst_cate}, Cold CATE p10: {cold_p10}, RV mean: {rv_mean}, Number of cold windows: {len(robust_cold)}"
                )

                slopes[int(name)] = {
                    "values": res["cate_linear"],
                    "capacity": res["control"].unique(),
                    "effect_min": res.groupby(["control"])["cate_linear"].min(),
                    "effect_max": res.groupby(["control"])["cate_linear"].max(),
                    "max_capacity": cap,
                    "cap_range": 0,
                    "worst_case_cate": worst_cate,
                    "cold_cate_p10": cold_p10,
                    "rv_mean": rv_mean,
                    "n_cold_windows": n_cold,
                    "feeder_size": feeder_size,
                }

        risk_data_list = []
        for fid, metrics in slopes.items():
            row = {
                "feeder": int(fid),
                "max_capacity": metrics["max_capacity"],
                "worst_case_cate": metrics["worst_case_cate"],
                "cold_cate_p10": metrics["cold_cate_p10"],
                "cap_range": metrics["cap_range"],
                "rv_mean": metrics["rv_mean"],
                "feeder_size": metrics["feeder_size"],
            }
            risk_data_list.append(row)

        risk_df = pd.DataFrame(risk_data_list)

        plot_risk_distribution(risk_df)
        plot_capacity_distribution(slopes)


# Smart Grids


def model_robustness(file_path, treatment, controls, dataset):

    robustness_configs = [
        {"name": "Baseline", "n_folds": 5, "algo": "lgbm", "interaction": False},
        {"name": "RF", "n_folds": 5, "algo": "rf", "interaction": False},
        {"name": "XGBoost", "n_folds": 5, "algo": "xgb", "interaction": False},
    ]

    all_test_results = []
    scores = {}
    for config in robustness_configs:
        print(f"--- Running {config['name']} ---")
        # Get the results_df which contains 'label' and 'cate_linear'
        label = "test_model" + config["name"]
        df = pd.read_csv(file_path)

        try:
            bool(datetime.strptime(df["Date"][0], "%Y-%m-%d"))
            df["month"] = pd.to_datetime(df["Date"], format="%Y-%m-%d").dt.month
            df["day"] = pd.to_datetime(df["Date"], format="%Y-%m-%d").dt.dayofweek
            df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour

        except:
            bool(datetime.strptime(df["Date"][0], "%d/%m/%Y"))
            df["month"] = pd.to_datetime(df["Date"], format="%d/%m/%Y").dt.month
            df["day"] = pd.to_datetime(df["Date"], format="%d/%m/%Y").dt.dayofweek
            df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour

        rmse_t, rmse_y, score, params_t, params_y = residualize_data(
            df,
            "Aggregate",
            controls,
            controls,
            treatment,
            config["n_folds"],
            config["algo"],
            algo_search=True,
        )

        scores[config["algo"]] = {
            "rmse_t": rmse_t,
            "rmse_y": rmse_y,
            "total": score,
            "params_t": params_t,
            "params_y": params_y,
        }
    master_df = pd.DataFrame(scores).T
    master_df.index.name = "algo"
    # Tag each row with the test namem
    # df_res['test_name'] = config['name']

    # Combine all results into one large DataFrame
    # master_df = pd.concat(all_test_results, ignore_index=True)
    master_df.to_csv(
        f"results/causality/{dataset}/master_robustness_results.csv", index=True
    )

    best_model = master_df["total"].idxmin()  # fix 4
    print(f"best model: {best_model}")
    print(f"  params_t: {master_df.loc[best_model, 'params_t']}")
    print(f"  params_y: {master_df.loc[best_model, 'params_y']}")


def causality_analysis(
    df,
    treatment,
    control_set,
    customer_size=None,
    dataset="elektro",
    n_iterations=1,
    save=True,
    sensitivity=True,
):

    # Scale the relevant columns
    try:
        bool(datetime.strptime(df["Date"][0], "%Y-%m-%d"))
        df["month"] = pd.to_datetime(df["Date"], format="%Y-%m-%d").dt.month
        df["day"] = pd.to_datetime(df["Date"], format="%Y-%m-%d").dt.dayofweek
        df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour

    except:
        bool(datetime.strptime(df["Date"][0], "%d/%m/%Y"))
        df["month"] = pd.to_datetime(df["Date"], format="%d/%m/%Y").dt.month
        df["day"] = pd.to_datetime(df["Date"], format="%d/%m/%Y").dt.dayofweek
        df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour

    # Group data by season, weekend vs weekday, night time vs morning vs afternoon vs evening
    df["day_type"] = df["day"].apply(lambda x: 0 if x < 5 else 1)

    df["label"] = df["day_type"].astype(str)
    eq = "y_res ~ t_res"

    treatment_possibilities = treatment
    window_size = 2000
    step_size = 200

    # load best model
    master = pd.read_csv(
        f"results/causality/{dataset}/master_robustness_results.csv", index_col="algo"
    )
    best = master["total"].idxmin()
    print(best)
    params_t = master.loc[best, "params_t"]
    params_y = master.loc[best, "params_y"]

    # plot_mean_data(df, label = f"{dataset}")

    for treatment in treatment_possibilities:
        controls = [c for c in control_set if c != treatment]
        print(f"Estimating CATE for treatment: {treatment} with controls: {controls}")

        available_stratums = sorted(df["label"].unique())

        df["agg"] = df["Aggregate"]
        results_df = pd.DataFrame()

        for m in available_stratums:
            if m == "0":
                window_size = 1000
                step_size = 100
            else:
                window_size = 500
                step_size = 50

            df_strat = df[df["label"] == m]

            for hour in range(24):
                hours_in_window = [(hour - 1) % 24, hour, (hour + 1) % 24]
                subset = df_strat[df_strat["hour"].isin(hours_in_window)]
                # Sort the DataFrame by trend
                subset = subset.sort_values(by=treatment)  # lowest to highest

                # Parameters for sliding window
                # Analysis is performed on overlapping windows to observe how effect changes over time
                # Store CATE and corresponding penetration levels for each window
                results = []

                # Total number of windows
                total_windows = (len(subset) - window_size) // step_size + 1

                # Sliding window analysis
                for start in range(
                    0, len(subset) - window_size + 1, step_size
                ):  # overlapping by 5000-500 = 4500 (0-4999, 500-5499)
                    window_data = subset.iloc[start : start + window_size]

                    # Calculate the mean trend for the current window
                    mean_temp = window_data[treatment].mean()

                    # Use tqdm for bootstrap iterations to estimate causal effect across different sample (build causal distribution)
                    # Capture non-stationary effects
                    with tqdm(
                        total=n_iterations,
                        desc=f"Window {start // step_size + 1}/{total_windows}",
                        leave=False,
                    ) as pbar:
                        # Bootstrap for the current window
                        for _ in range(n_iterations):
                            random_subset = window_data.sample(
                                n=min(len(window_data), window_size), replace=True
                            )
                            df_residualised = residualize_data(
                                random_subset,
                                "agg",
                                controls,
                                controls,
                                treatment,
                                5,
                                best,
                                params_t=params_t,
                                params_y=params_y,
                            )
                            # config['n_folds'], config['algo'], config['interaction'])

                            res = fit_residualized_model(df_residualised, controls, eq)

                            if sensitivity:
                                RV, RVA, b_res = sensitivity_analysis(
                                    num_splits=5,
                                    shuffle_data=True,
                                    shuffle_random_seed=42,
                                    benchmark_common_causes=["humidity"],
                                    significance_level=0.05,
                                    frac_strength_treatment=1,
                                    frac_strength_outcome=1,
                                    theta_s=res["cate_linear"],
                                    t_resid=df_residualised["t_resid"],
                                    agg_resid=df_residualised["agg_resid"],
                                    base_t=df_residualised["base_t"],
                                    base_y=df_residualised["base_y"],
                                    observed_common_causes=random_subset[controls],
                                    outcome=random_subset["agg"],
                                    treatment=random_subset[treatment],
                                    algo=best,
                                    params_t=params_t,
                                    params_y=params_y,
                                    plot=False,
                                )

                                # Append the mean trend value and CATE to results

                                results.append(
                                    {
                                        "label": m + "_" + str(hour),
                                        "control": hour,
                                        "mean_t": mean_temp,  # observe CATE across time
                                        "cate_linear": res[
                                            "cate_linear"
                                        ],  # Extract the first (and only) coefficient since there is only one feature (temp)
                                        "base_load": df_residualised["base_y"].mean(),
                                        "base_temp": df_residualised["base_t"].mean(),
                                        "load_res": df_residualised["agg_resid"].std(),
                                        "temp_res": df_residualised["t_resid"].std(),
                                        "rv_l": res["rv"],
                                        "rva_l": res["rva"],
                                        "partial_r2_l": res["partial_r2"],
                                        "rv": RV,
                                        "rva": RVA,
                                        "r2tu_w": b_res["r2tu_w"],
                                        "r2yu_tw": b_res["r2yu_tw"],
                                        "bias": b_res["bias"],
                                        "lower_ate_bound": b_res["lower_ate_bound"],
                                        "upper_ate_bound": b_res["upper_ate_bound"],
                                        "lower_confidence_bound": b_res[
                                            "lower_confidence_bound"
                                        ],
                                        "upper_confidence_bound": b_res[
                                            "upper_confidence_bound"
                                        ],
                                    }
                                )

                            else:
                                # Append the mean trend value and CATE to results
                                results.append(
                                    {
                                        "label": m + "_" + str(hour),
                                        "control": hour,
                                        "mean_t": mean_temp,  # observe CATE across time
                                        "cate_linear": res[
                                            "cate_linear"
                                        ],  # Extract the first (and only) coefficient since there is only one feature (temp)
                                        "base_load": df_residualised["base_y"].mean(),
                                        "base_temp": df_residualised["base_t"].mean(),
                                        "load_res": df_residualised["agg_resid"].std(),
                                        "temp_res": df_residualised["t_resid"].std(),
                                        "rv_l": res["rv"],
                                        "rva_l": res["rva"],
                                        "partial_r2_l": res["partial_r2"],
                                    }
                                )
                            pbar.update(1)  # Update the progress bar

                results_df = pd.concat(
                    [results_df, pd.DataFrame(results)], ignore_index=True
                )

        if save:
            results_df.to_csv(
                f"results/causality/{dataset}/results_causal_{treatment}_per_hour_day_{customer_size}.csv",
                index=False,
            )

            results_df = pd.read_csv(
                f"results/causality/{dataset}/results_causal_{treatment}_per_hour_day_{customer_size}.csv"
            )

            models, _, _ = get_guidance_function(results_df)

            label = treatment + "_" + str(customer_size)

            plot_CATE_with_sensitivity(results_df, dataset, feeder=0, label=label)
            plot_seasonal_hourly_cate(results_df, dataset, label)

            plot_seasonal_hourly_cate(results_df, dataset, label)

    # filepath= f"results_causal_{treatment}_per_hour_day_{customer_size}.csv"
    # df = pd.read_csv(filepath)
    # df['hour'] = df['control']

    # plot_robustness_distributions(df_all, dataset) #update so it is per hour

    return results_df


#
def estimate_CATE_control(df, f, conf_t, conf_y, treatment, outcome, control, eq):
    window_size = 3000
    step_size = 300
    n_iterations = 20  # Number of bootstraps in each window

    print(f"> {f}")
    # Sort the DataFrame by trend
    y = "outcome"
    df[y] = df[outcome]
    all_cap = df[control].unique()
    cap_numb = df[control].nunique()
    # print(cap_numb)
    # print(all_cap[0])
    results = []
    if not (all_cap[0] == 0.0 and cap_numb == 1):
        for cap in all_cap:
            # print(cap)
            df_cap = df[df[control] == cap]
            df_cap = df_cap.sort_values(
                by=treatment
            )  # destroys any temporal information here
            # print(len(df_cap))

            # Parameters for sliding window
            # Analysis is performed on overlapping windows to observe how effect changes over time
            # Store CATE and corresponding penetration levels for each window

            # Total number of windows
            total_windows = (len(df_cap) - window_size) // step_size + 1

            # Sliding window analysis
            for start in range(
                0, len(df_cap) - window_size + 1, step_size
            ):  # overlapping by 5000-500 = 4500 (0-4999, 500-5499)
                window_data = df_cap.iloc[start : start + window_size]

                # Calculate the mean trend for the current window
                mean_t = window_data[treatment].mean()
                # control = window_data['Heat_Pump_Capacity'].mean()

                # Use tqdm for bootstrap iterations to estimate causal effect across different sample (build causal distribution)
                # Capture non-stationary effects
                # with tqdm(total=n_iterations, desc=f'Window {start // step_size + 1}/{total_windows}', leave=False) as pbar:
                # Bootstrap for the current window
                for _ in range(n_iterations):
                    random_subset = window_data.sample(
                        n=min(len(window_data), window_size), replace=True
                    )
                    df_residualized = residualize_data(
                        random_subset, y, conf_t, conf_y, treatment
                    )
                    res = fit_residualized_model(df_residualized, conf_y, eq)

                    # Append the mean trend value and CATE to results
                    results.append(
                        {
                            "feeder": f,
                            "mean_t": mean_t,  # observe CATE across time
                            "control": cap,
                            "cate_linear": res[
                                "cate_linear"
                            ],  # Extract the first (and only) coefficient since there is only one feature (temp)
                            "rv": res["rv"],
                            "rva": res["rva"],
                            "partial_r2": res["partial_r2"],
                        }
                    )
                    # pbar.update(1)  # Update the progress bar

            results_df = pd.DataFrame(results)
    else:
        results_df = pd.DataFrame(results)
        # plot_CATE_time(results_df)
        # plot_CATE_with_sensitivity(results_df)
        # visualize_CATE_temporal(df, results_df, cate_lookup)
    results_df.to_csv(
        f"results/causality/uq_results_causal_temp_hp_{f}.csv", index=False
    )

    return results_df


def estimate_CATE_general(
    df, feeder, conf_t, conf_y, treatment, outcome, eq, type="nonlinear"
):
    window_size = 10000
    step_size = 1000
    n_iterations = 1  # Number of bootstraps in each window

    # Sort the DataFrame by trend
    y = "outcome"
    df[y] = df[outcome]
    df = df.sort_values(by=treatment)  # destroys any temporal information here

    # Parameters for sliding window
    # Analysis is performed on overlapping windows to observe how effect changes over time
    # Store CATE and corresponding penetration levels for each window
    results = []

    # Total number of windows
    total_windows = (len(df) - window_size) // step_size + 1

    # Sliding window analysis
    for start in range(
        0, len(df) - window_size + 1, step_size
    ):  # overlapping by 5000-500 = 4500 (0-4999, 500-5499)
        window_data = df.iloc[start : start + window_size]

        # Calculate the mean trend for the current window
        mean_t = window_data[treatment].mean()

        # Use tqdm for bootstrap iterations to estimate causal effect across different sample (build causal distribution)
        # Capture non-stationary effects
        with tqdm(
            total=n_iterations,
            desc=f"Window {start // step_size + 1}/{total_windows}",
            leave=False,
        ) as pbar:
            # Bootstrap for the current window
            for _ in range(n_iterations):
                random_subset = window_data.sample(
                    n=min(len(window_data), window_size), replace=True
                )
                df_residualized = residualize_data(
                    random_subset, y, conf_t, conf_y, treatment
                )
                res = fit_residualized_model(df_residualized, conf_y, eq, type)

                # Append the mean trend value and CATE to results
                results.append(
                    {
                        "mean_t": mean_t,  # observe CATE across time
                        "cate_linear": res[
                            "cate_linear"
                        ],  # Extract the first (and only) coefficient since there is only one feature (temp)
                        "rv": res["rv"],
                        "rva": res["rva"],
                        "partial_r2": res["partial_r2"],
                    }
                )
                pbar.update(1)  # Update the progress bar

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results/causality/results_causal_PV_{feeder}.csv", index=False)
    plot_CATE_with_sensitivity_temp(results_df)
    plot_CATE_time(results_df, feeder)

    return results_df


def refute_random_common_cause(
    dataset, num_simulations=50, random_state=42, run=True, df=None
):
    # generate random confounder
    random_state = np.random.RandomState(seed=random_state)

    # Load original data

    path = f"results/causality/{dataset}"
    og_df = pd.read_csv(
        f"{path}/results_causal_temperature_per_hour_day_1000_500.csv"
    )  # change name later

    if run:
        df = pd.read_csv(f"{df}")  # change name later
        treatment = ["temperature"]
        controls = [
            "temperature",
            "wind_speed",
            "humidity",
            "clear_sky_ghi",
            "rainfall",
            "month",
            "Holiday",
            "w_rand",
        ]
        sims = []

        for s in range(num_simulations):
            print(f"This is simulation {s} -----------------")
            new_df = df.assign(w_rand=random_state.normal(size=df.shape[0]))
            new_effect = causality_analysis(
                new_df,
                treatment,
                controls,
                dataset=dataset,
                save=False,
                sensitivity=False,
            )
            new_effect["sim_id"] = s
            sims.append(new_effect)

        new_estimates = pd.concat(sims, ignore_index=True)

        new_estimates.to_csv(
            f"results/causality/{dataset}/results_RCC_{num_simulations}.csv"
        )

    new_estimates = pd.read_csv(
        f"results/causality/{dataset}/results_RCC_{num_simulations}.csv"
    )

    sim_estimates = (
        new_estimates.groupby(["label", "mean_t", "sim_id"])["cate_linear"]
        .mean()
        .reset_index()
    )

    # original point estimate with bootstrap per (feature, label)
    og_stats = (
        og_df.groupby(["label", "mean_t"])[
            "cate_linear"
        ]  # not grouped by tempertaure where bootstraps are applied?
        .agg(orig_est="mean", orig_se="std")
        .reset_index()
    )

    rows = []
    for (label, t), grp in sim_estimates.groupby(["label", "mean_t"]):
        og_row = og_stats[(og_stats["label"] == label) & (og_stats["mean_t"] == t)]
        if og_row.empty:
            continue
        # Find original mean and std from bootstrap, should be scalar
        orig_est = og_row["orig_est"].iloc[0]
        orig_se = og_row["orig_se"].iloc[0]

        # Extract sim values
        sims_arr = grp[
            "cate_linear"
        ].values  # significance test needs the distribution of simulated estimates per window, not their average

        sig = test_significance(orig_est, sims_arr, significance_level=0.05)

        shift = abs(sims_arr.mean() - orig_est)
        rows.append(
            {
                "label": label,
                "mean_t": t,
                "orig_est": orig_est,
                "sim_mean": sims_arr.mean(),
                "sim_std": sims_arr.std(ddof=1),
                "rate_of_change": shift / max(abs(orig_est), 1e-8),
                "z_shift": shift / orig_se if orig_se and orig_se > 0 else np.nan,
                "p_value": sig["p_value"],
                # For RCC: significant difference = the random confounder MOVED
                # Refutation True means confounder did not move estimates, it is not significant means the effect was minimal >0.05
                "refutation_passed": not sig["is_statistically_significant"],
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(f"{path}/RCC_results.csv", index=False)

    results = pd.read_csv(f"{path}/RCC_results.csv")

    return results


def test_significance(estimate, simulations, significance_level=0.05):

    sims = np.asarray(simulations, dtype=float)
    num_simulations = len(sims)

    if num_simulations >= 100:  # Bootstrappin
        # Perform Bootstrap Significance Test with the original estimate and the set of refutations
        half_p = np.mean((sims > estimate) + 0.5 * (sims == estimate))
        p_value = 2 * min(half_p, 1 - half_p)

    else:
        # Perform Normal Tests of Significance with the original estimate and the set of refutations
        mu = sims.mean()
        sd = sims.std(ddof=1)
        if sd == 0:
            # Degenerate: all sims identical -> indistinguishable
            p_value = 1.0
            significance_dict = {
                "p_value": p_value,
                "is_statistically_significant": p_value <= significance_level,
            }
            return significance_dict

        z = (estimate - mu) / sd
        if z > 0:  # Right tail
            p_value = 1 - scipy.stats.norm.cdf(abs(z))
        else:
            p_value = scipy.stats.norm.cdf(abs(z))

    significance_dict = {
        "p_value": p_value,
        "is_statistically_significant": p_value <= significance_level,
    }

    return significance_dict


def filter_robust_estimates(df, dataset, run):
    # Filter the estimates based on the refutation results
    plot_CATE_with_sensitivity(df, dataset, feeder=0, label="og")
    plot_seasonal_hourly_cate(df, dataset, label="og")
    print(len(df))
    rcc_res = refute_random_common_cause(
        dataset, num_simulations=100, random_state=42, run=run
    )
    pass_rcc = rcc_res[rcc_res["refutation_passed"] == True]  # 31
    # Find how many are false
    print(len(rcc_res) - len(pass_rcc))
    # robust if the adjusted 95% confidence interval does not cross zero
    # both bounds are negative or possitive
    # SHOULD ALSO DELETE BASE LOAD AND BASE TEMP Rows
    df["is_passed"] = (df["lower_confidence_bound"].round(1) > 0) & (
        df["upper_confidence_bound"].round(1) > 0
    ) | (df["lower_confidence_bound"].round(1) < 0) & (
        df["upper_confidence_bound"].round(1) < 0
    )
    print(df[df["is_passed"] == False])
    pass_sens = df[df["is_passed"] == True]  # 32, 3
    print(len(df) - len(pass_sens))

    # merge
    pass_sens = pass_sens.merge(
        pass_rcc[["label", "mean_t"]], on=["label", "mean_t"], how="inner"
    )
    print(len(pass_sens))

    plot_CATE_with_sensitivity(pass_sens, dataset, feeder=0, label="filtered")
    plot_seasonal_hourly_cate(pass_sens, dataset, label="filtered")

    return pass_sens


def calculate_sub_impact(
    val_path, dataset, cal_path=None, task="val", selected_date=None
):
    cate_df = pd.read_csv(
        f"results/causality/{dataset}/results_causal_temperature_per_hour_day_hour.csv"
    )

    cate_df = filter_robust_estimates(cate_df, dataset, run=False)
    # Build the Interpolators
    cate_funcs, load_funcs, temp_funcs = get_guidance_function(cate_df)

    # Load Inference Data
    val_df = pd.read_csv(val_path)
    if task == "val":
        val_df.drop(
            columns=["target_load", "base_load", "upper_ci", "lower_ci"],
            inplace=True,
            errors="ignore",
        )
    else:
        val_df.drop(columns=["target_load", "base_load"], inplace=True, errors="ignore")

    # Convert dates to generate the matching Label keys
    dt = pd.to_datetime(val_df["Date"], format="mixed", dayfirst=True)

    dows = dt.dt.dayofweek
    hours = pd.to_datetime(val_df["Time"], format="%H:%M:%S").dt.hour

    # Helper to generate label
    def get_label(d, h):
        dtype = "0" if d < 5 else "1"

        return f"{dtype}_{h}"

    results = []

    # Use zip for faster iteration
    for t_curr, d, h in zip(val_df["temperature"], dows, hours):
        label = get_label(d, h)

        if label in cate_funcs:
            # Retrieve Physics for this specific Context & Temp
            cate_val = float(cate_funcs[label](t_curr))
            base_load_val = float(load_funcs[label](t_curr))
            base_temp_val = float(temp_funcs[label](t_curr))

            # Calculate Impact
            # Formula: Impact = Sensitivity * (Current_Temp - Normal_Temp_For_Context)
            impact = cate_val * (t_curr - base_temp_val)

            # Calculate Final Target
            pred = base_load_val + impact

            results.append(
                {
                    "target_load": pred,
                    "base_load": base_load_val,
                }
            )

    # Save Results
    res_df = pd.DataFrame(results)

    if task == "infer":
        val_df = pd.concat([val_df, res_df], axis=1).reset_index(drop=True)
        # Error
        # Calculate MAE
        mae = mean_absolute_error(val_df["target_load"], val_df["Aggregate"])
        print(f"MAE: {mae}")

        avg_load = val_df["Aggregate"].mean()
        print(avg_load)
        mape = (mae / avg_load) * 100
        print(f"Your Error %: {mape:.2f}%")
        mape = mean_absolute_percentage_error(
            val_df["target_load"], val_df["Aggregate"]
        )
        print(f"Your Error %: {mape:.2f}%")

        rmse = np.sqrt(mean_squared_error(val_df["target_load"], val_df["Aggregate"]))
        print(f"RMSE: {rmse}")

        if selected_date:
            val_df = val_df[
                (
                    (
                        pd.to_datetime(val_df["Date"], dayfirst=True)
                        >= str(selected_date[0])
                    )
                    & (
                        pd.to_datetime(val_df["Date"], dayfirst=True)
                        < str(selected_date[-1])
                    )
                )
            ].reset_index(drop=True)

        plt.figure(figsize=(12, 6))

        # Base Load
        plt.plot(
            val_df.index,
            val_df["base_load"],
            color="black",
            label="Base Load",
            alpha=0.6,
        )

        # Real Aggregate
        plt.plot(
            val_df.index, val_df["Aggregate"], color="red", label="Aggregate", alpha=0.6
        )

        # Prediction
        plt.plot(
            val_df.index,
            val_df["target_load"],
            color="blue",
            linewidth=1.5,
            label="Predicted Load",
        )

        # Styling
        plt.xlabel("Time (Half Hourly Steps)", fontsize=14)
        plt.ylabel("Load (kW)", fontsize=14)
        plt.title("2-Week Load Forecast: Base vs. Predicted", fontsize=14)
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"results/causality/{dataset}/causal_load_estimate_winter.png", dpi=300
        )
        # plt.show()
    else:
        val_df = pd.concat([val_df, res_df], axis=1).reset_index(drop=True)

    val_df.to_csv(val_path, index=False)
    return val_df


def get_guidance_function(results_df):
    models, base_load, base_temp = {}, {}, {}
    unique_labels = results_df["label"].unique()
    for label in unique_labels:
        # Extract data for this specific Season/Hour bucket
        subset = results_df[results_df["label"] == label].sort_values("mean_t")

        grouped = subset.groupby("mean_t")["cate_linear"].agg(["mean"])
        grouped["cate_lower_ci"] = subset.groupby("mean_t")["cate_linear"].quantile(
            0.05
        )
        grouped["cate_upper_ci"] = subset.groupby("mean_t")["cate_linear"].quantile(
            0.95
        )

        grouped["load_mean"] = subset.groupby("mean_t")["base_load"].agg(["mean"])
        grouped["load_lower_ci"] = subset.groupby("mean_t")["base_load"].quantile(0.05)
        grouped["load_upper_ci"] = subset.groupby("mean_t")["base_load"].quantile(0.95)

        grouped["temp_mean"] = subset.groupby("mean_t")["base_temp"].agg(["mean"])
        grouped["temp_lower_ci"] = subset.groupby("mean_t")["base_temp"].quantile(0.05)
        grouped["temp_upper_ci"] = subset.groupby("mean_t")["base_temp"].quantile(0.95)
        grouped = grouped.reset_index().sort_values("mean_t")

        # Create an interpolation function
        cate_func = interp1d(
            grouped["mean_t"],
            grouped["mean"],
            kind="linear",
            bounds_error=False,
            fill_value=(grouped["mean"].iloc[0], grouped["mean"].iloc[-1]),
        )
        baseload_func = interp1d(
            grouped["mean_t"],
            grouped["load_mean"],
            kind="linear",
            bounds_error=False,
            fill_value=(grouped["load_mean"].iloc[0], grouped["load_mean"].iloc[-1]),
        )
        basetemp_func = interp1d(
            grouped["mean_t"],
            grouped["temp_mean"],
            kind="linear",
            bounds_error=False,
            fill_value=(grouped["temp_mean"].iloc[0], grouped["temp_mean"].iloc[-1]),
        )

        models[label] = cate_func
        base_load[label] = baseload_func
        base_temp[label] = basetemp_func

    return models, base_load, base_temp


# removes effect of variables
# Remove confounders using LGBM to isolate treatment effect
def select_model(algo, params=None):
    params = params or {}
    if algo == "lgbm":
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            **params,
        )  # fix 3: string
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


def residualize_variable_kfold(
    data, y_var, x_vars, n_folds=5, algo="lgbm", params=None
):
    X = data[x_vars].values
    y = data[y_var].values

    if params is None:
        if algo == "lgbm":
            param = {
                "verbosity": [-1],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [5],  # Can model complex interactions (up to 5 vars)
                "num_leaves": [7, 15, 31],  # Allows for 15 distinct groups
                "min_child_samples": [
                    15,
                    40,
                    80,
                ],  # Very risky for N=1000, but sensitive to signal
                "reg_alpha": [0.1],  # Low regularization
                "subsample": [0.7, 0.9],
            }

        elif algo == "rf":
            param = {
                "min_samples_leaf": [5, 10, 20, 50],
                "max_features": [0.3, 0.5, 0.8, 1.0],
                "max_depth": [None, 6, 10],
            }

        elif algo == "xgb":
            param = {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 4, 6],
                "min_child_weight": [5, 15, 30],
                "subsample": [0.7, 0.9],
                "reg_lambda": [1.0, 5.0, 10.0],
            }

        search = RandomizedSearchCV(
            select_model(algo),
            param,
            n_iter=15,
            cv=5,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X, y.ravel())
        params = {
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in search.best_params_.items()
        }
    else:
        params = ast.literal_eval(params)

    kf = KFold(
        n_splits=n_folds, shuffle=True, random_state=42
    )  # cross validation for robustness

    residuals = np.zeros(len(y))
    base_feature = np.zeros(len(y))
    for train_idx, test_idx in kf.split(X):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = select_model(algo, params)  # why LCL needs ast.literal_eval(params)
        model.fit(X_train, y_train.ravel())

        # Predict and calculate residuals

        y_pred = model.predict(X_test)
        # substracts real from predicted to find the residual
        base_feature[test_idx] = y_pred

        residuals[test_idx] = y_test.ravel() - y_pred

    rmse_model = float(np.sqrt(np.mean(residuals**2)))

    return residuals, base_feature, rmse_model, params


# Residualize data
def residualize_data(
    data,
    y,
    conf_t,
    conf_y,
    treatment,
    n_folds=5,
    algo="lgbm",
    algo_search=False,
    params_t=None,
    params_y=None,
):
    if algo_search == False:
        # Purge signals by removing confounders
        # Find the influence of Time and humidity on temperature to separate its impact on temp
        data["t_resid"], data["base_t"], _, _ = residualize_variable_kfold(
            data, treatment, conf_t, n_folds=n_folds, algo=algo, params=params_t
        )

        # Find the influence of Time, day, month and irradiance on temperature to separate its impact on agg
        data["agg_resid"], data["base_y"], _, _ = residualize_variable_kfold(
            data, y, conf_y, n_folds=n_folds, algo=algo, params=params_y
        )

        return data

    else:
        _, _, rmse_t, params_t = residualize_variable_kfold(
            data, treatment, conf_t, n_folds=n_folds, algo=algo, params=params_t
        )

        # Find the influence of Time, day, month and irradiance on temperature to separate its impact on agg
        _, _, rmse_y, params_y = residualize_variable_kfold(
            data, y, conf_y, n_folds=n_folds, algo=algo, params=params_y
        )

        total = float(rmse_t / data[treatment[0]].std() + rmse_y / data[y].std())

        return rmse_t, rmse_y, total, params_t, params_y


# Standarises the residualised feature
def fit_residualized_model(data, conf, eq="y_res ~ t_res"):
    scaler = StandardScaler()
    X_residualized = data[["t_resid"]]
    y_residualized = data["agg_resid"]
    predictions = {}

    pipeline_residualized = Pipeline(
        [("scaler", scaler), ("regressor", LinearRegression())]
    )

    pipeline_residualized.fit(X_residualized, y_residualized)
    coef_residualized = pipeline_residualized.named_steps["regressor"].coef_
    scale_residualized = pipeline_residualized.named_steps[
        "scaler"
    ].scale_  # find 1 std dev, change in load per 1 SD in treatment
    predictions[f"cate_linear"] = coef_residualized / scale_residualized
    predictions[f"cate_linear"] = predictions[f"cate_linear"][0]

    # test robustness at this stage

    X = data[conf].values

    residual_df = pd.DataFrame(
        {
            "y_res": data[["agg_resid"]]
            .reset_index(drop=True)
            .iloc[:, 0],  # Should average across split indexes
            "t_res": X_residualized.reset_index(drop=True).iloc[:, 0],
        }
    )

    ols_model = smf.ols(eq, data=residual_df).fit()

    if "t_res" not in ols_model.bse or ols_model.bse["t_res"] == 0:
        return {
            "cate_linear": predictions.get("cate_linear", 0),
            "rv": np.nan,
            "rva": np.nan,
            "partial_r2": 0,
        }

    try:
        sens = smkr.Sensemakr(model=ols_model, treatment="t_res")

        print(sens.sensitivity_stats)
        predictions["rv"] = sens.sensitivity_stats["rv_q"]
        predictions["rva"] = sens.sensitivity_stats["rv_qa"]
        predictions["partial_r2"] = sens.sensitivity_stats["r2yd_x"]
    # predictions['bounds'] = sens.bounds  # full DataFrame if you want kd breakdown
    except Exception as e:
        print(f"Sensemakr failed: {e}")
        predictions["rv"], predictions["rva"], predictions["partial_r2"] = (
            np.nan,
            np.nan,
            0,
        )

    return predictions
