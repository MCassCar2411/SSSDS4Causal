import random

import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances

from process_dataset import *


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def median_heuristic_gamma(X):
    # Calculate all pairwise Euclidean distances
    dists = euclidean_distances(X, X, squared=True)
    # Find the median of the non-zero distances
    median_sq_dist = np.median(dists[dists > 0])
    # Calculate gamma
    gamma = 1.0 / median_sq_dist
    return gamma


def mmd(n_samples, real_data, synthetic_data, level="agg", seeds=4, gamma=1.0, seed=42):
    print(f"Running MMD level: {level}")
    # Convert DataLoader to numpy if needed
    if isinstance(real_data, torch.Tensor):
        real_data = real_data.detach().cpu().numpy()
    if isinstance(synthetic_data, torch.Tensor):
        synthetic_data = synthetic_data.detach().cpu().numpy()

    if level == "agg":
        real = real_data  # already (B, T)
        synth = synthetic_data.sum(axis=-1)  # (B, T)
        real = real / (real.max(axis=1, keepdims=True) + 1e-8)
        synth = synth / (synth.max(axis=1, keepdims=True) + 1e-8)
    else:
        real = real_data.transpose(0, 2, 1).reshape(-1, real_data.shape[1])  # (B*C, T)
        synth = synthetic_data.transpose(0, 2, 1).reshape(-1, synthetic_data.shape[1])

        real = real / (real.max(axis=1, keepdims=True) + 1e-8)
        synth = synth / (synth.max(axis=1, keepdims=True) + 1e-8)

    print(f"Real: {real.shape}, Synthetic: {synth.shape}")

    n = min(len(real), len(synth), n_samples)
    rng = np.random.default_rng(seed)
    # real = real[rng.choice(len(real), n, replace=False)]
    # synth = synth[rng.choice(len(synth), n, replace=False)]

    gamma_sample = real[rng.choice(len(real), min(500, len(real)), replace=False)]

    gamma = median_heuristic_gamma(gamma_sample)
    print(f"This is gamma {gamma}")

    if np.isnan(gamma) or gamma == 0:
        print("Warning: gamma is NaN or 0, falling back to gamma=1.0")
        gamma = 1.0

    rng = np.random.default_rng(seed)
    scores = []
    # aggregate doesn't need this really
    for i in range(seeds):
        real_idx = rng.choice(len(real), size=n, replace=False)
        synth_idx = rng.choice(len(synth), size=n, replace=False)
        real_subset = real[real_idx]
        synth_subset = synth[synth_idx]

        xx = metrics.pairwise.rbf_kernel(real_subset, real_subset, gamma)
        yy = metrics.pairwise.rbf_kernel(synth_subset, synth_subset, gamma)
        xy = metrics.pairwise.rbf_kernel(real_subset, synth_subset, gamma)

        score = xx.mean() + yy.mean() - 2 * xy.mean()
        scores.append(score)
        print(f"Seed: {i}\tScore: {score:.4f}")

    mean = np.mean(scores)
    std = np.std(scores)
    print(f"MMD Score: {mean:.4f}, {std:.4f}")
    return mean, std


def wd(n_samples, real_dataset, generated_dataset, level="agg", seeds=4, seed=42):
    print(f"Running WD level: {level}")

    # Convert to numpy
    if isinstance(real_dataset, torch.Tensor):
        real_dataset = real_dataset.detach().cpu().numpy()
    if isinstance(generated_dataset, torch.Tensor):
        generated_dataset = generated_dataset.detach().cpu().numpy()

    if level == "agg":
        # (B, T, C) -> (B, T) aggregate
        real = real_dataset  # assume already aggregated
        synth = generated_dataset.sum(axis=-1)
        # normalise per day
        real = real / (real.max(axis=1, keepdims=True) + 1e-8)
        synth = synth / (synth.max(axis=1, keepdims=True) + 1e-8)
    else:
        # (B, T, C) -> (B*C, T) individual household-days
        real = real_dataset.transpose(0, 2, 1).reshape(-1, real_dataset.shape[1])
        synth = generated_dataset.transpose(0, 2, 1).reshape(
            -1, generated_dataset.shape[1]
        )
        # normalise per household-day
        real = real / (real.max(axis=1, keepdims=True) + 1e-8)
        synth = synth / (synth.max(axis=1, keepdims=True) + 1e-8)

    print(f"Real: {real.shape}, Synthetic: {synth.shape}")

    all_scores = []
    rng = np.random.default_rng(seed)
    n = min(len(real), len(synth), n_samples)

    for s in range(seeds):
        # Independent sampling for real and synthetic
        real_idx = rng.choice(len(real), size=n, replace=False)
        synth_idx = rng.choice(len(synth), size=n, replace=False)
        real_subset = real[real_idx]  # (n, T)
        synth_subset = synth[synth_idx]  # (n, T)

        # WD per time-step dimension across samples (distributional comparison)
        # wd = wasserstein_distance_nd(real_subset, synth_subset)
        wd = wasserstein_distance(real_subset.flatten(), synth_subset.flatten())
        # mean_wd = np.mean(wd_list)
        all_scores.append(wd)
        print(f"Seed: {s}\tWD Score: {str(wd).replace('.', ',')}")

    mean = np.mean(all_scores)
    std = np.std(all_scores)
    return mean, std


def rmse_mae_score(
    real_dataset, generated_dataset, level="agg", n_samples=500, seeds=4, seed=42
):
    print(f"Running RMSE/MAE level: {level}")

    # Convert to numpy
    if isinstance(real_dataset, torch.Tensor):
        real_dataset = real_dataset.detach().cpu().numpy()
    if isinstance(generated_dataset, torch.Tensor):
        generated_dataset = generated_dataset.detach().cpu().numpy()

    if level == "agg":
        # (B, T, C) -> (B, T) aggregate
        real = real_dataset  # assume already aggregated
        synth = generated_dataset.sum(axis=-1)
        # normalise per day
        real = real / (real.max(axis=1, keepdims=True) + 1e-8)
        synth = synth / (synth.max(axis=1, keepdims=True) + 1e-8)
    else:
        # (B, T, C) -> (B*C, T) individual household-days
        real = real_dataset.transpose(0, 2, 1).reshape(-1, real_dataset.shape[1])
        synth = generated_dataset.transpose(0, 2, 1).reshape(
            -1, generated_dataset.shape[1]
        )
        # normalise per household-day
        real = real / (real.max(axis=1, keepdims=True) + 1e-8)
        synth = synth / (synth.max(axis=1, keepdims=True) + 1e-8)

    print(f"Real: {real.shape}, Synthetic: {synth.shape}")

    rmse_list, mae_list = [], []
    rng = np.random.default_rng(seed)
    n = min(len(real), len(synth), n_samples)

    for i in range(seeds):
        # Independent sampling � not paired unless data is reconstructive
        real_idx = rng.choice(len(real), size=n, replace=False)
        synth_idx = rng.choice(len(synth), size=n, replace=False)
        real_subset = real[real_idx]
        synth_subset = synth[synth_idx]

        # Compare distribution means/stds per timestep rather than paired pointwise
        real_mean = real_subset.mean(axis=0)  # (T,)
        synth_mean = synth_subset.mean(axis=0)  # (T,)
        real_std = real_subset.std(axis=0)
        synth_std = synth_subset.std(axis=0)

        rmse = np.sqrt(mean_squared_error(real_mean, synth_mean))
        mae = mean_absolute_error(real_mean, synth_mean)
        rmse_list.append(rmse)
        mae_list.append(mae)
        print(
            f"Seed: {i}\tRMSE: {str(rmse).replace('.', ',')}\tMAE: {str(mae).replace('.', ',')}"
        )

    return {
        "rmse_mean": np.mean(rmse_list),
        "rmse_std": np.std(rmse_list),
        "mae_mean": np.mean(mae_list),
        "mae_std": np.std(mae_list),
    }


def compute_power_stats(data, label, level="household"):
    results = {}
    # B, T, C = data.shape

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if level == "aggregate":
        if label == "synth":
            data = data.sum(axis=2, keepdims=False)  # aggregate over customers (B, T)

        data_hourly = data.reshape(data.shape[0], 24, 2).mean(axis=2)

        results = {
            "overall_mean": data.mean(),
            "overall_peak": data.max(),
            "daily_mean": data.mean(axis=1),  # (B)
            "daily_peak": data.max(axis=1),  # (B)
            "hourly_mean": data.mean(axis=0),  # (48)
            "hourly_peak": data.max(axis=0),  # (48)
        }

        print(f"\n{label}, {level} Power Stats")
        print("Overall Mean:", round(results["overall_mean"], 3))
        print("Overall Peak:", round(results["overall_peak"], 3))
        print("Half Hourly Mean Profile:", np.round(results["hourly_mean"], 2))
        print("Half Hourly Peak Profile:", np.round(results["hourly_peak"], 2))
        print("Daily Mean Avg:", np.round(results["daily_mean"].mean(), 2))
        print("Daily Peak Avg:", np.round(results["daily_peak"].mean(), 2))

    else:
        data_hourly = data.reshape(data.shape[0], 24, 2, data.shape[2]).mean(
            axis=2
        )  # (B, 24, C)

        results = {
            "overall_mean": data.mean(),
            "overall_peak": data.max(),
            "daily_mean": data.mean(axis=1),  # (B, C)
            "daily_peak": data.max(axis=1),  # (B, C)
            "hourly_mean": data.mean(axis=0),  # (48, C)
            "hourly_peak": data.max(axis=0),  # (48, C)
        }

        print(f"\n{label}, {level} Power Stats")
        print("Overall Mean:", round(results["overall_mean"], 3))
        print("Overall Peak:", round(results["overall_peak"], 3))
        print("Half Hourly Mean Profile:", np.round(results["hourly_mean"], 2))
        print("Half Hourly Peak Profile:", np.round(results["hourly_peak"], 2))
        print("Daily Mean Avg:", np.round(results["daily_mean"].mean(), 2))
        print("Daily Peak Avg:", np.round(results["daily_peak"].mean(), 2))

    return results


def power_stat_distances(stats_real, stats_gen):
    distances = {
        "mean_diff_overall": abs(
            stats_real["overall_mean"] - stats_gen["overall_mean"]
        ),
        "peak_diff_overall": abs(
            stats_real["overall_peak"] - stats_gen["overall_peak"]
        ),
        "hourly_mean_rmse": np.sqrt(
            ((stats_real["hourly_mean"] - stats_gen["hourly_mean"]) ** 2).mean()
        ),
        "hourly_peak_rmse": np.sqrt(
            ((stats_real["hourly_peak"] - stats_gen["hourly_peak"]) ** 2).mean()
        ),
        "daily_mean_rmse": np.sqrt(
            ((stats_real["daily_mean"] - stats_gen["daily_mean"]) ** 2).mean()
        ),
        "daily_peak_rmse": np.sqrt(
            ((stats_real["daily_peak"] - stats_gen["daily_peak"]) ** 2).mean()
        ),
    }
    return distances


def compute_admd_mdmd(dataset, label, C):
    if label == "synth":
        group_demand = dataset.sum(axis=2)  # sum over customers -> (B, S)

        max_demand = group_demand.max(axis=1)  # max over time -> (B,)
        admd = max_demand / C  # divide by num_customers
        # max diversified demand
    else:
        max_demand = dataset.max(axis=1)  # max over time -> (B,)
        admd = max_demand / C  # divide by num_customers

    return admd


def normalise_metrics(all_metrics):
    # Collect all metric names
    metric_names = ["mmd", "wd", "rmse", "mae"]

    # Flatten and collect all values per metric
    all_values_per_metric = {metric: [] for metric in metric_names}
    for loss_name in all_metrics:
        for metric in metric_names:
            all_values_per_metric[metric].extend(all_metrics[loss_name][metric])

    # Compute global min and max for each metric
    metric_min = {
        metric: np.min(values) for metric, values in all_values_per_metric.items()
    }
    metric_max = {
        metric: np.max(values) for metric, values in all_values_per_metric.items()
    }

    # Normalize all values in the original dictionary
    normalized_metrics = {}
    for loss_name in all_metrics:
        normalized_metrics[loss_name] = {}
        for metric in metric_names:
            values = np.array(all_metrics[loss_name][metric])
            min_val = metric_min[metric]
            max_val = metric_max[metric]
            # Avoid division by zero
            if max_val - min_val == 0:
                norm_values = np.zeros_like(values)
            else:
                norm_values = (values - min_val) / (max_val - min_val)
            normalized_metrics[loss_name][metric] = norm_values

    return normalized_metrics


def measure_peaks(data, C=158, label="real"):
    if label == "synth":
        data = data.sum(axis=2)  # B, T

        max_demand = data.max(axis=1)  # max over time -> (B,)
        admd = max_demand / C  # divide by num_customers
        # max diversified demand
    else:
        max_demand = data.max(axis=1)  # max over time -> (B,)
        admd = max_demand / C  # divide by num_customers

    # Find cardinal points
    sigma = 1
    night_end = 13  # 6am
    day_end = 25  # 12pm
    eve_start = 29  # 2pm
    n_days = len(data)
    d1 = gaussian_filter1d(data, sigma=sigma, order=1, axis=1)
    d2 = gaussian_filter1d(data, sigma=sigma, order=2, axis=1)

    n_days, n_steps = data.shape
    data = gaussian_filter1d(data, sigma=sigma, order=0, axis=1)

    night_min_idx = np.argmin(
        data[:, 0:night_end], axis=1
    )  # window starts at 0 -> already absolute

    morning_max_idx = np.empty(n_days, dtype=int)
    for i in range(n_days):
        start = night_min_idx[i]
        window = data[i, start:day_end]
        morning_max_idx[i] = start + np.argmax(window)

    evening_max_idx = np.empty(n_days, dtype=int)
    for i in range(n_days):
        start = eve_start  # after 2pm
        window = data[i, start:]
        evening_max_idx[i] = start + np.argmax(window)

    day_trough_idx = np.empty(n_days, dtype=int)
    for i in range(n_days):
        start = morning_max_idx[i]
        end = evening_max_idx[i]
        window = data[i, start:end]
        day_trough_idx[i] = start + np.argmin(window)

    # Nightime fall
    fs_night = np.empty(n_days)
    for i in range(n_days):
        if night_min_idx[i] == 0:
            fs_night[i] = 0
        else:
            fs_night[i] = np.mean(d1[i, 0 : night_min_idx[i]])
    fs_night_duration = night_min_idx.astype(float)
    # Morning rising slope
    rs_morning = np.empty(n_days)

    for i in range(n_days):
        start = night_min_idx[i]
        end = morning_max_idx[i]
        rs_morning[i] = np.mean(d1[i, start:end])

    rs_morning_duration = morning_max_idx - night_min_idx

    # Night-to-day curvature
    nighttoday_curvature = np.empty(n_days)
    for i in range(n_days):
        start = 0
        end = morning_max_idx[i]
        window = d2[i, start:end]
        nighttoday_curvature[i] = np.mean(np.abs(window))

    # Evening rising slope: window [daytime_trough_idx, end), per row ---
    rs_evening = np.empty(n_days)
    for i in range(n_days):
        start = day_trough_idx[i]
        end = evening_max_idx[i]
        rs_evening[i] = np.mean(d1[i, start:end])

    rs_evening_duration = evening_max_idx - day_trough_idx

    # Evening falling slope: window [rs_evening_idx, end), per row ---
    fs_evening = np.empty(n_days)
    for i in range(n_days):
        start = evening_max_idx[i]
        fs_evening[i] = np.mean(d1[i, start:])

    fs_evening_duration = n_steps - evening_max_idx

    # --- Evening-to-night curvature: peak |curvature| over [rs_evening_idx, end) ---
    evetonight_curvature = np.empty(n_days)
    for i in range(n_days):
        start = evening_max_idx[i]
        evetonight_curvature[i] = np.mean(np.abs(d2[i, start:]))

    agg_feat = {}
    agg_feat = {
        "admd": admd,
        "fall slope night": fs_night,  # fall slope min of 1st deriv
        "fall slope night duration": fs_night_duration,  # timesteps
        "rise slope morning": rs_morning,  # rise slope max of 1st deriv
        "rise slope morning duration": rs_morning_duration,
        "Morning curvature": nighttoday_curvature,  # second deriv
        "rise slope evening": rs_evening,
        "rise slope evening duration": rs_evening_duration,
        "fall slope evening": fs_evening,
        "fall slope evening duration": fs_evening_duration,
        "Evening curvature": evetonight_curvature,  # second deriv
    }
    print(agg_feat)

    return agg_feat
