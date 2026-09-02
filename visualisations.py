import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE


def plot_tsne(
    real_data,
    synthetic_data,
    labels,
    label,
    level="agg",
    perplexity=30,
    num_samples=500,
    seed=42,
):

    if isinstance(synthetic_data, torch.Tensor):
        synthetic_data = synthetic_data.detach().cpu().numpy()
    if isinstance(real_data, torch.Tensor):
        real_data = real_data.detach().cpu().numpy()

    if level == "agg":
        # Sum to get agg
        real_data_agg = real_data
        synthetic_data_agg = synthetic_data.sum(axis=-1)

        # normalise per day
        real_data_agg = real_data_agg / real_data_agg.max(axis=1, keepdims=True)
        synthetic_data_agg = synthetic_data_agg / synthetic_data_agg.max(
            axis=1, keepdims=True
        )

        print(f"Real: {real_data_agg.shape}, Synthetic: {synthetic_data_agg.shape}")

        n_real = len(real_data_agg)
        combined = np.vstack([real_data_agg, synthetic_data_agg])

        tsne = TSNE(
            n_components=2,
            perplexity=15,
            random_state=seed,
            n_iter=300,
            early_exaggeration=10.0,
            init="pca",
            learning_rate="auto",
        )
        embedding = tsne.fit_transform(combined)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            embedding[0:n_real][:, 0],
            embedding[0:n_real][:, 1],  # real_x and real_y
            c="mediumblue",
            alpha=0.66,
            s=30,
            label=labels[0],
        )
        plt.scatter(
            embedding[n_real:][:, 0],
            embedding[n_real:][:, 1],  # gen_x and gen_y
            c="firebrick",
            alpha=0.66,
            s=30,
            label=labels[1],
            marker="x",
        )
        # plt.title(f"t-SNE: Real vs Synthetic ({label})")
        plt.xlabel("t-SNE Component 1", fontsize=18)
        plt.ylabel("t-SNE Component 2", fontsize=18)
        plt.tick_params(axis="both", labelsize=18)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(f"results/tsne_{label}_agg.png", dpi=300)
        plt.close()

    else:
        # plot each individual customer = BxC, T
        real_data_ind = real_data.transpose(0, 2, 1)  # (195, 158, 48)
        real_data_ind = real_data_ind.reshape(-1, 48)  # (30810, 48)

        synthetic_data_ind = synthetic_data.transpose(0, 2, 1)  # (195, 158, 48)
        synthetic_data_ind = synthetic_data_ind.reshape(-1, 48)  # (30810, 48)

        # normalise per day
        real_data_ind = real_data_ind / (
            real_data_ind.max(axis=1, keepdims=True) + 1e-8
        )
        synthetic_data_ind = synthetic_data_ind / (
            synthetic_data_ind.max(axis=1, keepdims=True) + 1e-8
        )

        rng = np.random.default_rng(seed)

        real_data_ind = real_data_ind[
            rng.choice(len(real_data_ind), num_samples, replace=False)
        ]
        synthetic_data_ind = synthetic_data_ind[
            rng.choice(len(synthetic_data_ind), num_samples, replace=False)
        ]

        print(f"Real: {real_data_ind.shape}, Synthetic: {synthetic_data_ind.shape}")

        n_real = num_samples
        combined = np.vstack([real_data_ind, synthetic_data_ind])

        tsne = TSNE(
            n_components=2,
            perplexity=15,
            random_state=seed,
            n_iter=300,
            early_exaggeration=10.0,
            init="pca",
            learning_rate="auto",
        )
        embedding = tsne.fit_transform(combined)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            embedding[0:n_real][:, 0],
            embedding[0:n_real][:, 1],  # real_x and real_y
            c="mediumblue",
            alpha=0.66,
            s=30,
            label=labels[0],
        )
        plt.scatter(
            embedding[n_real:][:, 0],
            embedding[n_real:][:, 1],  # gen_x and gen_y
            c="firebrick",
            alpha=0.66,
            s=30,
            label=labels[1],
            marker="x",
        )
        # plt.title(f"t-SNE: Real vs Synthetic ({label})")
        plt.xlabel("t-SNE Component 1", fontsize=16)
        plt.ylabel("t-SNE Component 2", fontsize=16)
        plt.tick_params(axis="both", labelsize=16)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(f"results/tsne_{label}_ind.png", dpi=300)
        plt.close()


def plot_aug_synthetic_data(
    aggregate, ind, x0, n_samples, input_dim, label, mean_aggregate
):
    print(aggregate.shape)
    print(np.shape(x0))

    # Calculate the mean of the concatenated aggregate values
    _, ax = plt.subplots(figsize=(12, 6))

    # Plot each individual sample run in grey
    for i in range(n_samples):
        (_,) = ax.plot(aggregate[i, :], color="grey", alpha=0.3, linewidth=1)

    # Plot Ground Truth

    (line_real,) = ax.plot(
        x0.flatten().tolist(),
        color="red",
        linestyle="--",
        linewidth=2,
        label="Real Dataset Demand",
    )

    # Plot Mean of Samples
    (line_mean,) = ax.plot(
        mean_aggregate, color="blue", linewidth=2, label="Mean Synthetic Demand"
    )

    # Formatting
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Time Steps (30 min)")
    ax.set_title(f"Aggregate Load: {n_samples} Samples vs Real Data ({label})")

    # Proxy artist for the grey lines in legend
    from matplotlib.lines import Line2D

    custom_lines = [Line2D([0], [0], color="grey", lw=1), line_real, line_mean]
    ax.legend(
        custom_lines,
        [f"Synthetic Samples (n={n_samples})", "Real Ground Truth", "Mean Synthetic"],
        loc="upper right",
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/aug_profiles_agg_comparison_{label}.png")
    plt.show()

    # Individual Customer (First Customer Example) ---
    _, ax = plt.subplots(figsize=(12, 6))
    for i in range(4):
        customer_lines = ax.plot(ind[0, :, i], color="blue", alpha=0.2)

    custom_lines = [Line2D([0], [0], color="blue", lw=1), customer_lines]  # pyright: ignore[reportPossiblyUnboundVariable]
    ax.legend(custom_lines, ["Synthetic Samples for 4 customers"], loc="upper right")
    ax.set_title("Individual Customer Demand For a Single Sample")
    plt.grid(True, alpha=0.3)
    plt.show()
    # Save and show the plot
    plt.savefig(f"results/aug_profiles_ind_comparison_{input_dim}.png")


def plot_mean_data(data, label, seed=None):
    # Compute aggregate across features (customers)
    # B, T, C
    loss_names = list(data.keys())

    num_batches = data[loss_names[0]]["real"].shape[0]
    rng = np.random.default_rng(seed)
    # Select a random batch index to represent a random day
    random_batch_index = rng.integers(low=0, high=num_batches)
    print(random_batch_index)

    _, ax = plt.subplots(figsize=(10, 5))
    for loss_name in data:
        aggregate_s = data[loss_name]["synth"][random_batch_index, :, :].sum(axis=1)
        ax.plot(
            aggregate_s, label=f"Synthetic Aggregate Demand {loss_name}", linewidth=2
        )

    aggregate_r = data[loss_names[0]]["real"][random_batch_index, :, :].sum(axis=1)
    ax.plot(aggregate_r, label="Real Aggregate Demand", linewidth=2, color="red")
    ax.set_ylabel("Power (kW)", fontsize=12)
    ax.set_xlabel("Time Steps (30 min)", fontsize=12)
    ax.set_title("Generated Synthetic Aggregate Load", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/mean_profiles_agg_{label}.png")
    plt.close()

    num_customers = data[loss_names[0]]["real"].shape[2]

    # Select a random customer index
    random_customer_index = rng.integers(low=0, high=num_customers)
    print(random_customer_index)
    _, ax = plt.subplots(figsize=(10, 5))
    for loss_name in data:
        aggregate_s = data[loss_name]["synth"][
            random_batch_index, :, random_customer_index
        ]
        ax.plot(
            aggregate_s, label=f"Synthetic Aggregate Demand {loss_name}", linewidth=2
        )

    aggregate_r = data[loss_names[0]]["real"][
        random_batch_index, :, random_customer_index
    ]
    ax.plot(aggregate_r, label="Real Aggregate Demand", linewidth=2, color="red")
    ax.set_ylabel("Power (kW)", fontsize=12)
    ax.set_xlabel("Time Steps (30 min)", fontsize=12)
    ax.set_title("Generated Single Customer Synthetic Load", fontsize=14)
    ax.set_ylabel("Power (kW)", fontsize=12)
    ax.set_xlabel("Time Steps (30 min)", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/mean_profiles_single_{label}.png")
    plt.close()


def plot_training_curve(train_losses, loss_name, input_dim):
    """Plots the training and validation loss over epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", marker="o")
    # plt.plot(val_losses, label="Validation Loss", marker="o", linestyle="dashed")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()
    plt.savefig(f"results/train_curve_{loss_name}_{input_dim}.png")
    plt.close()


def plot_metric_line_chart(all_metrics, label):
    metric_names = ["mmd", "wd"]
    levels = ["agg", "ind"]
    level_titles = {"agg": "Aggregate Level", "ind": "Individual Level"}
    p_values = list(all_metrics.keys())

    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, level in zip(axes, levels):
        for metric in metric_names:
            values = [all_metrics[p][level][metric] for p in p_values]
            ax.plot(p_values, values, marker="o", label=metric.upper())
        ax.set_xlabel("P Strength")
        ax.set_ylabel("Score")
        ax.set_title(f"{level_titles[level]} - Scores Across P Values - {label}")
        ax.set_xticks(p_values)
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.savefig(f"results/p_ablation_{label}.png", dpi=150)
    plt.close()


def plot_metric_bar_chart(all_metrics, input_dim):
    metric_names = ["wd", "rmse", "mae", "mmd"]
    loss_names = list(all_metrics.keys())
    levels = ["agg", "ind"]
    level_titles = {"agg": "Aggregate Level", "ind": "Individual Level"}

    num_metrics = len(metric_names)
    num_losses = len(loss_names)
    x = np.arange(num_losses)
    bar_width = 0.2

    _, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, level in zip(axes, levels):
        for i, metric in enumerate(metric_names):
            metric_values = [all_metrics[loss][level][metric] for loss in loss_names]
            ax.bar(
                x + i * bar_width, metric_values, width=bar_width, label=metric.upper()
            )
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title(f"{level_titles[level]} - {input_dim}")
        ax.set_xticks(x + bar_width * (num_metrics - 1) / 2)
        ax.set_xticklabels(loss_names, rotation=45)
        ax.legend()
        ax.grid(axis="y")

    plt.tight_layout()
    plt.savefig(f"results/all_metrics_grouped_bar_chart_{input_dim}.png", dpi=150)
    plt.close()


def plot_admd_mdmd_distributions(admd_stats, input_dim):
    loss_names = list(admd_stats.keys())
    colors = ("dodgerblue", "forestgreen", "firebrick", "darkorange", "purple")
    plt.figure(figsize=(10, 6))

    admd_real = admd_stats[loss_names[0]]["real"].flatten()
    plt.hist(
        admd_real,
        bins=50,
        density=True,
        histtype="step",
        linewidth=3,
        linestyle="-",
        color="k",
        label="Real",
    )
    for color, loss_name in zip(colors, loss_names):
        admd_gen = admd_stats[loss_name]["generated"].flatten()

        plt.hist(
            admd_gen,
            bins=50,
            density=True,
            histtype="step",
            linewidth=2,
            linestyle="-",
            color=color,
            label=f"{loss_name}",
        )

        score = wasserstein_distance(admd_real, admd_gen)
        print(
            f"Loss Function: {loss_name}\t WD Score for ADMD: {str(score).replace('.', ',')}"
        )

    plt.xlabel("ADMD Value", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.tick_params(axis="both", labelsize=18)
    # plt.title(f"ADMD Distribution Comparison Across Models — {input_dim}")
    plt.legend(fontsize=18, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/admd_dist_all_models_{input_dim}.png", dpi=300)
    plt.close()


def plot_all_testing_results(all_metrics, admd_stats, agg_features, input_dim):
    os.makedirs("results", exist_ok=True)
    plot_metric_bar_chart(all_metrics, input_dim)
    plot_admd_mdmd_distributions(admd_stats, input_dim)
    plot_deriv_distirbutions(agg_features, input_dim)


def plot_deriv_distirbutions(aggregate_feat, label):
    loss_names = list(aggregate_feat.keys())
    real = aggregate_feat[loss_names[0]]["real"]

    metrics = list(real.keys())

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.2 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, metric in zip(axes, metrics):
        sns.kdeplot(
            np.asarray(real[metric]).flatten(), label="Real", linewidth=2, ax=ax
        )
        for loss in loss_names:
            synth_vals = np.asarray(aggregate_feat[loss]["synth"][metric]).flatten()
            std = np.std(synth_vals)
            if std < 1e-8:
                ax.axvline(synth_vals[0], linewidth=2, label=loss)

            sns.kdeplot(synth_vals, label=loss, linewidth=2, ax=ax)

        ax.set_yscale("log")
        ax.set_xlabel(metric)
        ax.set_ylabel("Density")
        ax.set_title(metric)
        ax.grid(True, alpha=0.4)

    # hide any unused subplot axes (when n_metrics is odd)
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=len(labels_))
    fig.suptitle(f"Feature Distributions — {label}", y=1.0, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(f"results/feature_dist_{label}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Wasserstein distance
    wd_scores = {loss: {} for loss in loss_names}
    for loss in loss_names:
        synth = aggregate_feat[loss]["synth"]
        for metric in metrics:
            real_vals = np.asarray(real[metric]).flatten()
            synth_vals = np.asarray(synth[metric]).flatten()
            if metric == "rise slope evening":
                print(real_vals)
                print(synth_vals)
            wd_scores[loss][metric] = wasserstein_distance(real_vals, synth_vals)

    # WD summary bar chart
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * n_metrics), 6))
    x = np.arange(n_metrics)
    width = 0.8 / len(loss_names)
    colors = ("dodgerblue", "forestgreen", "firebrick", "darkorange", "purple")

    for i, (loss, color) in enumerate(zip(loss_names, colors)):
        vals = [wd_scores[loss][metric] for metric in metrics]
        ax.bar(
            x + i * width - 0.4 + width / 2, vals, width=width, color=color, label=loss
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=14)
    ax.set_ylabel("Wasserstein Distance", fontsize=16)
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=14)
    # ax.set_title(f"Feature WD Summary — {label}")
    ax.grid(True, alpha=0.4, axis="y")
    ax.legend(
        loc="upper center",
        ncol=len(loss_names),
        bbox_to_anchor=(0.5, 1.15),
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(f"results/feature_wd_summary_{label}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return wd_scores


# delete later
def plot_peaks(data_synth, data_real, label):
    data_synth = data_synth.sum(axis=2)  # B, T

    # OG
    _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data_synth.flatten(), label="Synthetic", linewidth=2)

    ax.plot(data_real.flatten(), label="Real", linewidth=2, color="red")
    ax.set_ylabel("Power (kW)", fontsize=16)
    ax.set_xlabel("Time Steps (30 min)", fontsize=16)
    ax.legend(loc="upper right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/comparison_{label}.png")


def plot_power_stats_by_loss(all_real_stats, all_gen_stats, power_metrics, input_dim):

    loss_names = list(power_metrics.keys())

    for level in ["household", "aggregate"]:
        # Line Plot: Hourly Mean Profiles
        plt.figure(figsize=(10, 6))
        # for loss in loss_names:
        #   if level == 'aggregate':
        #      plt.plot(np.arange(48), all_gen_stats[loss][level]['hourly_mean'], label=f"{loss} (gen)")

        if level == "aggregate":
            plt.plot(
                np.arange(48),
                all_real_stats[loss_names[0]][level]["hourly_mean"],
                label="Real",
                linestyle="-",
                color="black",
            )

        plt.title(f"Half Hourly Mean Profile - {input_dim}")
        plt.xlabel("Half Hour")
        plt.ylabel("kW")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"results/hourly_mean_profile_{level}_{input_dim}.png")
        plt.close()

        # Boxplot: Daily Mean Distributions
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        data = [
            all_gen_stats[loss][level]["daily_mean"].flatten() for loss in loss_names
        ]
        plt.boxplot(
            data + [all_real_stats[loss_names[0]][level]["daily_mean"].flatten()],
            tick_labels=loss_names + ["Real"],
        )
        plt.title(f"Daily Mean Distribution - {level}, {input_dim}")
        plt.ylabel("kW")
        plt.xticks(rotation=45)
        plt.grid()

        # Density plot
        labels = loss_names + ["Real"]
        plt.subplot(2, 1, 2)
        for i, d in enumerate(
            data + [all_real_stats[loss_names[0]][level]["daily_mean"].flatten()]
        ):
            sns.kdeplot(d, label=labels[i], linewidth=2)

        # plt.title(f'Daily Mean Distribution - {level} (Density)')
        plt.xlabel("kW")
        plt.ylabel("Density")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig(f"results/daily_mean_dist_{level}_{input_dim}.png")
        plt.close()

        # Daily Peak
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)

        data = [
            all_gen_stats[loss][level]["daily_peak"].flatten() for loss in loss_names
        ]
        plt.boxplot(
            data + [all_real_stats[loss_names[0]][level]["daily_peak"].flatten()],
            tick_labels=loss_names + ["Real"],
        )
        plt.title(f"Daily Peak Distribution - {level}, {input_dim}")
        plt.ylabel("kW")
        plt.xticks(rotation=45)
        plt.grid()

        # Density plot
        plt.subplot(2, 1, 2)

        for i, d in enumerate(
            data + [all_real_stats[loss_names[0]][level]["daily_peak"].flatten()]
        ):
            sns.kdeplot(d, label=labels[i], linewidth=2)

        # plt.title(f'Daily Peak Distribution - {level} (Density)')
        plt.xlabel("kW")
        plt.ylabel("Density")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig(f"results/daily_peak_dist_{level}_{input_dim}.png")
        plt.close()

        # Boxplot: Hourly Mean Distributions
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        data = [
            all_gen_stats[loss][level]["hourly_mean"].flatten() for loss in loss_names
        ]
        plt.boxplot(
            data + [all_real_stats[loss_names[0]][level]["hourly_mean"].flatten()],
            tick_labels=loss_names + ["Real"],
        )
        plt.title(f"Half Hourly Mean Distribution - {level}, {input_dim}")
        plt.ylabel("kW")
        plt.xticks(rotation=45)
        plt.grid()

        # Density plot
        plt.subplot(2, 1, 2)

        for i, d in enumerate(
            data + [all_real_stats[loss_names[0]][level]["hourly_mean"].flatten()]
        ):
            sns.kdeplot(d, label=labels[i], linewidth=2)

        # plt.title(f'Daily Peak Distribution - {level} (Density)')
        plt.xlabel("kW")
        plt.ylabel("Density")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"results/hourly_mean_dist_{level}_{input_dim}.png")
        plt.close()

        # Boxplot: Hourly Peak Distributions
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        data = [
            all_gen_stats[loss][level]["hourly_peak"].flatten() for loss in loss_names
        ]
        plt.boxplot(
            data + [all_real_stats[loss_names[0]][level]["hourly_peak"].flatten()],
            tick_labels=loss_names + ["Real"],
        )
        plt.title(f"Half Hourly Peak Distribution - {level}, {input_dim}")
        plt.ylabel("kW")
        plt.xticks(rotation=45)
        plt.grid()

        plt.subplot(2, 1, 2)

        for i, d in enumerate(
            data + [all_real_stats[loss_names[0]][level]["hourly_peak"].flatten()]
        ):
            sns.kdeplot(d, label=labels[i], linewidth=2)

        # plt.title(f'Daily Peak Distribution - {level} (Density)')
        plt.xlabel("kW")
        plt.ylabel("Density")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig(f"results/hourly_peak_dist_{level}_{input_dim}.png")
        plt.close()


def plot_hourly_error_boxplots(power_values, input_dim):

    loss_names = list(power_values.keys())
    half_hours = np.arange(24)
    levels = ["aggregate", "household"]

    n_rows, n_cols = len(loss_names), len(levels)
    _, axes = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows), sharex=True, sharey=False
    )

    # Normalize axes to a consistent 2D array regardless of n_rows/n_cols
    axes = np.atleast_2d(axes)
    if n_rows == 1 and n_cols > 1:
        axes = axes.reshape(1, n_cols)
    elif n_cols == 1 and n_rows > 1:
        axes = axes.reshape(n_rows, 1)

    for r, loss in enumerate(loss_names):
        for c, level in enumerate(levels):
            ax = axes[r, c]

            real_arr = power_values[loss]["real"]
            gen_arr = power_values[loss]["synth"]

            if level == "aggregate":
                # Sum over households -> (48, C)
                real_stat = (
                    real_arr.sum(axis=2, keepdims=False)
                    .reshape(real_arr.shape[0], 24, 2)
                    .mean(axis=2)
                )
                gen_stat = (
                    gen_arr.sum(axis=2, keepdims=False)
                    .reshape(gen_arr.shape[0], 24, 2)
                    .mean(axis=2)
                )
            else:  # household
                # Keep per-household resolution: flatten (days, households) per half-hour
                real_stat = (
                    real_arr.reshape(real_arr.shape[0], -1)
                    if real_arr.shape[0] == 24
                    else real_arr.reshape(-1, real_arr.shape[-1]).T
                )
                gen_stat = (
                    gen_arr.reshape(gen_arr.shape[0], -1)
                    if gen_arr.shape[0] == 24
                    else gen_arr.reshape(-1, gen_arr.shape[-1]).T
                )

            error = gen_stat - real_stat

            if error.ndim == 1:
                box_data = [[error[t]] for t in half_hours]
            else:
                if error.shape[0] != 24:
                    error = error.T  # -> (48, n)
                box_data = [error[t] for t in half_hours]

            ax.boxplot(
                box_data,
                positions=half_hours,
                widths=0.6,
                patch_artist=True,
                boxprops={"facecolor": "steelblue", "alpha": 0.6},
                medianprops={"color": "red", "linewidth": 1.5},
                flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
                whiskerprops={"linewidth": 1},
                capprops={"linewidth": 1},
            )
            # ax.axhline(0, color='black', linewidth=1.2, linestyle='--', label='Zero error')
            ax.set_title(f"{loss} — {level.capitalize()}", fontsize=16)
            if level == "aggregate":
                ax.set_ylabel("Error (kW)", fontsize=18)
            ax.grid(axis="y", alpha=0.4)
            ax.tick_params(axis="y", labelsize=18)
            # ax.legend(fontsize=8)

    for c in range(n_cols):
        # axes[-1, c].set_xlabel('Half-Hour')
        axes[-1, c].set_xticks(half_hours[::2])
        axes[-1, c].set_xticklabels(
            [f"{h:02d}:00" for h in half_hours[::2]], rotation=45, fontsize=18
        )

    plt.tight_layout()
    plt.savefig(
        f"results/hourly_error_boxplot_{input_dim}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


# Causality plots
# Control split CATE plot
def plot_seasonal_hourly_cate(results_df, dataset, label):

    results_df["control"] = results_df["control"].astype(float)
    results_df["rv"] = results_df["rv"].astype(float)

    caps = sorted(results_df["control"].unique())

    results_df[["day_type", "hour_str"]] = results_df["label"].str.split(
        "_", expand=True
    )

    unique_day = [d for d in results_df["day_type"].unique()]

    fig, axes = plt.subplots(
        1,
        len(unique_day),
        figsize=(8, 5),
        constrained_layout=True,
        sharey=True,
        dpi=300,
    )
    # Define Color Map for Hours (0-23)
    cmap = plt.get_cmap("turbo")
    norm = mcolors.Normalize(
        vmin=min(caps) if len(caps) > 1 else 0, vmax=max(caps) if len(caps) > 1 else 10
    )

    for i, day in enumerate(unique_day):
        day_data = results_df[results_df["day_type"] == day]

        for c, colour in zip(caps, [cmap(norm(c)) for c in caps]):
            df_subset = day_data[day_data["control"] == c]
            grouped_cate = (
                df_subset.groupby("mean_t")["cate_linear"].agg(["mean"]).reset_index()
            )
            grouped_cate["lower_ci"] = (
                df_subset.groupby("mean_t")["cate_linear"].quantile(0.05).values
            )
            grouped_cate["upper_ci"] = (
                df_subset.groupby("mean_t")["cate_linear"].quantile(0.95).values
            )
            grouped_cate = grouped_cate.sort_values("mean_t")

            # Plotting (Use smoothed_mean now)
            axes[i].plot(
                grouped_cate["mean"],
                grouped_cate["mean_t"],
                color=colour,
                alpha=0.8,
                linewidth=1.8,
                label=f"{c} kW",
            )
            axes[i].fill_betweenx(
                grouped_cate["mean_t"],
                grouped_cate["lower_ci"],
                grouped_cate["upper_ci"],
                color=colour,
                alpha=0.2,
            )

        # Formats and Label Management
        axes[i].set_xlabel("CATE (kW/°C)", fontsize=14)
        axes[0].set_ylabel("Temperature (°C)", fontsize=14)
        axes[i].axvline(0, color="black", linestyle="-", linewidth=1.5, alpha=0.4)
        axes[i].grid(True, alpha=0.25)
        axes[i].tick_params(axis="both", which="major", labelsize=14)
        axes[i].tick_params(axis="both", which="minor", labelsize=14)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.05, shrink=0.7
    )
    cbar.set_ticklabels(["Midnight", "5 AM", "10 AM", "5 PM", "10 PM"], fontsize=12)

    # Save
    plt.savefig(
        f"results/causality/{dataset}/Capacity_CATE_Profiles_{label}.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_CATE_with_sensitivity(results_df, dataset, feeder, label):
    # Create the figure with an extra column for the colorbar
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    # Define common mapping parameters
    cmap = "RdYlBu_r"  # Red for warm, Blue for cold

    mean_cate_temp = (
        results_df.groupby("mean_t")["cate_linear"].agg(["mean"]).reset_index()
    )
    mean_cate_temp["lower_ci"] = (
        results_df.groupby("mean_t")["cate_linear"].quantile(0.1).values
    )
    mean_cate_temp["upper_ci"] = (
        results_df.groupby("mean_t")["cate_linear"].quantile(0.9).values
    )
    mean_cate_temp["control"] = (
        results_df.groupby("mean_t")["control"].mean().values
    )  # Round for cleaner x-axis labels

    mean_rv_temp = results_df.groupby("mean_t")["rv"].agg(["mean"]).reset_index()
    mean_rv_temp["rva"] = results_df.groupby("mean_t")["rva"].mean().values
    mean_rv_temp["lower_ci_rv"] = (
        results_df.groupby("mean_t")["rv"].quantile(0.1).values
    )
    mean_rv_temp["upper_ci_rv"] = (
        results_df.groupby("mean_t")["rv"].quantile(0.9).values
    )
    mean_rv_temp["control"] = (
        results_df.groupby("mean_t")["control"].mean().values
    )  # Round for cleaner x-axis labels

    mean_r2_temp = results_df.groupby("mean_t")["r2yu_tw"].agg(["mean"]).reset_index()
    mean_r2_temp["lower_ci_r2"] = (
        results_df.groupby("mean_t")["r2yu_tw"].quantile(0.1).values
    )
    mean_r2_temp["upper_ci_r2"] = (
        results_df.groupby("mean_t")["r2yu_tw"].quantile(0.9).values
    )
    mean_r2_temp["control"] = (
        results_df.groupby("mean_t")["control"].mean().values
    )  # Round for cleaner x-axis labels

    # Panel 1: CATE estimate
    sc1 = axes[0].scatter(
        mean_cate_temp["mean"],
        mean_cate_temp["control"],
        c=mean_cate_temp["mean_t"],
        cmap=cmap,
        edgecolors="k",
        alpha=0.7,
        label="Est. CATE",
    )

    # add_trendline(axes[0], results_df['effect_min'], capacity_10, 'blue')
    axes[0].set_xlabel("CATE (kW/°C)")
    axes[0].set_ylabel("Hour of day")
    axes[0].set_title("Estimated CATE")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Robustness Value (RV)
    axes[1].scatter(
        mean_rv_temp["mean"],
        mean_cate_temp["control"],
        c=mean_rv_temp["mean_t"],
        cmap=cmap,
        edgecolors="k",
        alpha=0.7,
        label="RV (q=1)",
    )
    # Also adding the alpha-level RV as smaller dots to keep visual focus
    axes[1].scatter(
        mean_rv_temp["rva"],
        mean_cate_temp["control"],
        c=mean_rv_temp["mean_t"],
        cmap=cmap,
        marker="x",
        s=20,
        alpha=0.5,
        label="RV (α=0.05)",
    )

    # axes[1].axvline(0.1, color='orange', linestyle=':', label='RV=0.1 (fragile)')
    axes[1].set_xlabel("Robustness Value")
    axes[1].set_title("Sensitivity: Robustness Value")
    axes[1].legend(loc="lower right", fontsize="small")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Partial R2
    axes[2].scatter(
        mean_r2_temp["mean"],
        mean_r2_temp["control"],
        c=mean_r2_temp["mean_t"],
        cmap=cmap,
        edgecolors="k",
        alpha=0.7,
    )

    axes[2].set_xlabel("Partial R2 of Treatment")
    axes[2].set_title("Treatment Explanatory Power")
    axes[2].grid(True, alpha=0.3)

    # Add Colorbar
    plt.subplots_adjust(right=0.9)  # Make room for colorbar
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))  # [left, bottom, width, height]
    cbar = fig.colorbar(sc1, cax=cbar_ax)
    cbar.set_label(f"Mean {label}", rotation=270, labelpad=15)

    plt.suptitle(
        "CATE and Sensitivity Analysis (Color = Mean Temp Window)", fontsize=14, y=0.98
    )
    plt.savefig(f"results/causality/{dataset}/CATE_{feeder}_{label}.png")
    plt.close()
