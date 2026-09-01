from tkinter import font
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
import os
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import wasserstein_distance
import random
from scipy.ndimage import gaussian_filter1d
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm


def plot_tsne(real_data, synthetic_data, labels,
              label, level = 'agg', perplexity=30, num_samples=500, seed=42):

    if isinstance(synthetic_data, torch.Tensor):
        synthetic_data = synthetic_data.detach().cpu().numpy()
    if isinstance(real_data, torch.Tensor):
        real_data = real_data.detach().cpu().numpy()


    if level == 'agg':
        # Sum to get agg
        real_data_agg = real_data
        synthetic_data_agg = synthetic_data.sum(axis=-1)

        #normalise per day
        real_data_agg = real_data_agg / real_data_agg.max(axis=1, keepdims=True)
        synthetic_data_agg = synthetic_data_agg / synthetic_data_agg.max(axis=1, keepdims=True)


        print(f"Real: {real_data_agg.shape}, Synthetic: {synthetic_data_agg.shape}")

        n_real = len(real_data_agg)
        combined = np.vstack([real_data_agg, synthetic_data_agg])

        tsne = TSNE(n_components=2, perplexity=15, random_state=seed, n_iter=300, early_exaggeration=10.0,
                    init='pca', learning_rate='auto')
        embedding = tsne.fit_transform(combined)

        plt.figure(figsize=(8, 6))
        plt.scatter(embedding[0:n_real][:, 0], embedding[0:n_real][:, 1], #real_x and real_y
                    c='mediumblue', alpha=0.66, s=30, label=labels[0])
        plt.scatter(embedding[n_real:][:, 0], embedding[n_real:][:, 1], #gen_x and gen_y
                    c='firebrick', alpha=0.66, s=30, label=labels[1], marker='x')
        #plt.title(f"t-SNE: Real vs Synthetic ({label})")
        plt.xlabel("t-SNE Component 1", fontsize=18)
        plt.ylabel("t-SNE Component 2",  fontsize=18)
        plt.tick_params(axis='both', labelsize=18)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(f"results/tsne_{label}_agg.png", dpi=300)
        plt.close()

    else:

        #plot each individual customer = BxC, T
        real_data_ind = real_data.transpose(0, 2, 1)  # (195, 158, 48)
        real_data_ind = real_data_ind.reshape(-1, 48)      # (30810, 48)

        synthetic_data_ind = synthetic_data.transpose(0, 2, 1)  # (195, 158, 48)
        synthetic_data_ind = synthetic_data_ind.reshape(-1, 48)      # (30810, 48)

        #normalise per day
        real_data_ind = real_data_ind / (real_data_ind.max(axis=1, keepdims=True)+1e-8)
        synthetic_data_ind = synthetic_data_ind / (synthetic_data_ind.max(axis=1, keepdims=True)+1e-8)

        rng = np.random.default_rng(seed)

        real_data_ind = real_data_ind[rng.choice(len(real_data_ind), num_samples, replace=False)]
        synthetic_data_ind = synthetic_data_ind[rng.choice(len(synthetic_data_ind), num_samples, replace=False)]


        print(f"Real: {real_data_ind.shape}, Synthetic: {synthetic_data_ind.shape}")

        n_real = num_samples
        combined = np.vstack([real_data_ind, synthetic_data_ind])
        #breakpoint()
        tsne = TSNE(n_components=2, perplexity=15, random_state=seed, n_iter=300, early_exaggeration=10.0,
                    init='pca', learning_rate='auto')
        embedding = tsne.fit_transform(combined)

        plt.figure(figsize=(8, 6))
        plt.scatter(embedding[0:n_real][:, 0], embedding[0:n_real][:, 1], #real_x and real_y
                    c='mediumblue', alpha=0.66, s=30, label=labels[0])
        plt.scatter(embedding[n_real:][:, 0], embedding[n_real:][:, 1], #gen_x and gen_y
                    c='firebrick', alpha=0.66, s=30, label=labels[1], marker='x')
        #plt.title(f"t-SNE: Real vs Synthetic ({label})")
        plt.xlabel("t-SNE Component 1", fontsize=16)
        plt.ylabel("t-SNE Component 2",  fontsize=16)
        plt.tick_params(axis='both', labelsize=16)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(f"results/tsne_{label}_ind.png", dpi=300)
        plt.close()


def plot_aug_synthetic_data(aggregate, ind, x0, n_samples, input_dim, label, mean_aggregate):
    print(aggregate.shape)
    print(np.shape(x0))

    # Calculate the mean of the concatenated aggregate values
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each individual sample run in grey
    for i in range(n_samples):
        line_grey, = ax.plot(aggregate[i, :], color="grey", alpha=0.3, linewidth=1)
    
    # Plot Ground Truth
    
    line_real, = ax.plot(x0.flatten().tolist()  , color="red", linestyle="--", linewidth=2, label="Real Dataset Demand")
    
    # Plot Mean of Samples
    line_mean, = ax.plot(mean_aggregate, color="blue", linewidth=2, label="Mean Synthetic Demand")
    
    # Formatting
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Time Steps (30 min)")
    ax.set_title(f"Aggregate Load: {n_samples} Samples vs Real Data ({label})")
    
    # Proxy artist for the grey lines in legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color="grey", lw=1), line_real, line_mean]
    ax.legend(custom_lines, [f"Synthetic Samples (n={n_samples})", "Real Ground Truth", "Mean Synthetic"], loc="upper right")
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/aug_profiles_agg_comparison_{label}.png")
    plt.show()

    # Individual Customer (First Customer Example) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    customer_idx = 0
    n_customers = ind.shape[2]
    for i in range(4):
        customer_lines = ax.plot(ind[0, :, i], color="blue", alpha=0.2)
    
    # Note: If x0 only contains aggregate, you might need individual ground truth here
    # Assuming mean_customers was calculated
    mean_ind = ind[:, :, customer_idx].mean(axis=0)
    #ax.plot(mean_ind, color="blue", linewidth=2, label="Mean Synthetic Customer")
    custom_lines = [Line2D([0], [0], color="blue", lw=1), customer_lines]
    ax.legend(custom_lines, [f"Synthetic Samples for 4 customers"], loc="upper right")
    ax.set_title(f"Individual Customer Demand For a Single Sample")
    plt.grid(True, alpha=0.3)
    plt.show()
    # Save and show the plot
    plt.savefig(f"results/aug_profiles_ind_comparison_{input_dim}.png")


def plot_mean_data(data, label, seed=None):
    """
    plt.figure(figsize=(10, 6))
    print(data.shape)
 
    plt.plot(np.arange(48),  data.groupby(['Time'])['Aggregate'].mean(), label="Historical Mean", linestyle='-', color='black') 

    #plt.title(f'Half Hourly Mean Profile - {input_dim}')
    plt.xlabel('Time steps (30 min)')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"results/hourly_mean_profile_{label}.png")
    plt.close()
    """
    # Compute aggregate across features (customers)
    #B, T, C
    loss_names = list(data.keys())

    num_batches = data[loss_names[0]]['real'].shape[0]
    rng = np.random.default_rng(seed)
    # Select a random batch index to represent a random day
    random_batch_index = rng.integers(low=0, high=num_batches)
    print(random_batch_index)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    for loss_name in data:
        aggregate_s = data[loss_name]['synth'][random_batch_index, :, :].sum(axis=1)
        ax.plot(aggregate_s, label=f"Synthetic Aggregate Demand {loss_name}", linewidth=2)

    aggregate_r = data[loss_names[0]]['real'][random_batch_index, :, :].sum(axis=1)
    ax.plot(aggregate_r, label="Real Aggregate Demand", linewidth=2, color="red")
    ax.set_ylabel("Power (kW)", fontsize=12)
    ax.set_xlabel("Time Steps (30 min)", fontsize=12)
    ax.set_title("Generated Synthetic Aggregate Load", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/mean_profiles_agg_{label}.png")
    plt.close()

    num_customers = data[loss_names[0]]['real'].shape[2]
    
    # Select a random customer index
    random_customer_index = rng.integers(low=0, high=num_customers)
    print(random_customer_index)
    fig, ax = plt.subplots(figsize=(10, 5))
    for loss_name in data:
        aggregate_s = data[loss_name]['synth'][random_batch_index, :, random_customer_index]
        ax.plot(aggregate_s, label=f"Synthetic Aggregate Demand {loss_name}", linewidth=2)
    
    aggregate_r = data[loss_names[0]]['real'][random_batch_index, :, random_customer_index]
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
    #plt.plot(val_losses, label="Validation Loss", marker="o", linestyle="dashed")
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, level in zip(axes, levels):
        for i, metric in enumerate(metric_names):
            metric_values = [all_metrics[loss][level][metric] for loss in loss_names]
            ax.bar(x + i * bar_width, metric_values, width=bar_width, label=metric.upper())
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title(f"{level_titles[level]} - {input_dim}")
        ax.set_xticks(x + bar_width * (num_metrics - 1) / 2)
        ax.set_xticklabels(loss_names, rotation=45)
        ax.legend()
        ax.grid(axis='y')

    plt.tight_layout()
    plt.savefig(f"results/all_metrics_grouped_bar_chart_{input_dim}.png", dpi=150)
    plt.close()


def plot_admd_mdmd_distributions(admd_stats, input_dim):
    loss_names = list(admd_stats.keys())
    colors = ("dodgerblue", "forestgreen", "firebrick", "darkorange", "purple")
    plt.figure(figsize=(10, 6))

    admd_real = admd_stats[loss_names[0]]['real'].flatten()
    plt.hist(admd_real, bins=50, density=True, histtype='step',
                  linewidth=3, linestyle='-', color='k',
                  label=f"Real")
    for color, loss_name in zip(colors, loss_names):
       
        admd_gen = admd_stats[loss_name]['generated'].flatten()

        
        plt.hist(admd_gen, bins=50, density=True, histtype='step',
                  linewidth=2, linestyle='-', color=color,
                  label=f"{loss_name}")

        score = wasserstein_distance(admd_real, admd_gen)
        print(f'Loss Function: {loss_name}\t WD Score for ADMD: {str(score).replace(".", ",")}')

    plt.xlabel("ADMD Value", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.tick_params(axis='both', labelsize=18)
    #plt.title(f"ADMD Distribution Comparison Across Models — {input_dim}")
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
    real = aggregate_feat[loss_names[0]]['real']

    metrics = list(real.keys())

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.2 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, metric in zip(axes, metrics):
        
        sns.kdeplot(np.asarray(real[metric]).flatten(), label="Real", linewidth=2, ax=ax)
        for loss in loss_names:
           synth_vals = np.asarray(aggregate_feat[loss]['synth'][metric]).flatten()
           std = np.std(synth_vals)
           if std < 1e-8:
               ax.axvline(synth_vals[0], linewidth=2,
                           label=loss)

           sns.kdeplot(synth_vals, label=loss, linewidth=2, ax=ax)

        ax.set_yscale('log')
        ax.set_xlabel(metric)
        ax.set_ylabel("Density")
        ax.set_title(metric)
        ax.grid(True, alpha=0.4)

    # hide any unused subplot axes (when n_metrics is odd)
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper center', ncol=len(labels_))
    fig.suptitle(f"Feature Distributions — {label}", y=1.0, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"results/feature_dist_{label}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Wasserstein distance 
    wd_scores = {loss: {} for loss in loss_names}
    for loss in loss_names:
        synth = aggregate_feat[loss]['synth']
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
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, color=color, label=loss)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha='right', fontsize=14)
    ax.set_ylabel("Wasserstein Distance", fontsize=16)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    #ax.set_title(f"Feature WD Summary — {label}")
    ax.grid(True, alpha=0.4, axis='y')
    ax.legend(loc='upper center', ncol=len(loss_names), bbox_to_anchor=(0.5, 1.15), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(f"results/feature_wd_summary_{label}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    return wd_scores

#delete later
def plot_peaks(data_synth, data_real, label):
    data_synth = data_synth.sum(axis=2) #B, T
    
    #OG 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data_synth.flatten(), label=f"Synthetic", linewidth=2)
   
    ax.plot(data_real.flatten(), label="Real", linewidth=2, color="red")
    ax.set_ylabel("Power (kW)", fontsize=16)
    ax.set_xlabel("Time Steps (30 min)", fontsize=16)
    ax.legend(loc="upper right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/comparison_{label}.png")

def plot_power_stats_by_loss(all_real_stats, all_gen_stats, power_metrics, input_dim):

   
    loss_names = list(power_metrics.keys())

    for level in ['household', 'aggregate']:

        # Line Plot: Hourly Mean Profiles
        plt.figure(figsize=(10, 6))
        #for loss in loss_names:
         #   if level == 'aggregate':
          #      plt.plot(np.arange(48), all_gen_stats[loss][level]['hourly_mean'], label=f"{loss} (gen)") 

        if level == 'aggregate':
            plt.plot(np.arange(48), all_real_stats[loss_names[0]][level]['hourly_mean'], label="Real", linestyle='-', color='black') 


        plt.title(f'Half Hourly Mean Profile - {input_dim}')
        plt.xlabel('Half Hour')
        plt.ylabel('kW')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"results/hourly_mean_profile_{level}_{input_dim}.png")
        plt.close()

        # Boxplot: Daily Mean Distributions
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        data = [all_gen_stats[loss][level]['daily_mean'].flatten() for loss in loss_names]
        plt.boxplot(data + [all_real_stats[loss_names[0]][level]['daily_mean'].flatten()], labels=loss_names + ['Real'])
        plt.title(f'Daily Mean Distribution - {level}, {input_dim}')
        plt.ylabel('kW')
        plt.xticks(rotation=45)
        plt.grid()

        # Density plot
        labels=loss_names + ['Real']
        plt.subplot(2, 1, 2)
        for i, d in enumerate(data + [all_real_stats[loss_names[0]][level]['daily_mean'].flatten()]):
            sns.kdeplot(d, label=labels[i], linewidth=2)

        #plt.title(f'Daily Mean Distribution - {level} (Density)')
        plt.xlabel('kW')
        plt.ylabel('Density')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig(f"results/daily_mean_dist_{level}_{input_dim}.png")
        plt.close()


        #Daily Peak
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)

        data = [all_gen_stats[loss][level]['daily_peak'].flatten() for loss in loss_names]
        plt.boxplot(data + [all_real_stats[loss_names[0]][level]['daily_peak'].flatten()], labels=loss_names + ['Real'])
        plt.title(f'Daily Peak Distribution - {level}, {input_dim}')
        plt.ylabel('kW')
        plt.xticks(rotation=45)
        plt.grid()
        

        #Density plot
        plt.subplot(2, 1, 2)

        for i, d in enumerate(data + [all_real_stats[loss_names[0]][level]['daily_peak'].flatten()]):
            sns.kdeplot(d, label=labels[i], linewidth=2)

        #plt.title(f'Daily Peak Distribution - {level} (Density)')
        plt.xlabel('kW')
        plt.ylabel('Density')
        plt.legend()
        plt.grid()


        plt.tight_layout()
        plt.savefig(f"results/daily_peak_dist_{level}_{input_dim}.png")
        plt.close()

        # Boxplot: Hourly Mean Distributions
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        data = [all_gen_stats[loss][level]['hourly_mean'].flatten() for loss in loss_names]
        plt.boxplot(data + [all_real_stats[loss_names[0]][level]['hourly_mean'].flatten()], labels=loss_names + ['Real'])
        plt.title(f'Half Hourly Mean Distribution - {level}, {input_dim}')
        plt.ylabel('kW')
        plt.xticks(rotation=45)
        plt.grid()

        #Density plot
        plt.subplot(2, 1, 2)

        for i, d in enumerate(data + [all_real_stats[loss_names[0]][level]['hourly_mean'].flatten()]):
            sns.kdeplot(d, label=labels[i], linewidth=2)

        

        #plt.title(f'Daily Peak Distribution - {level} (Density)')
        plt.xlabel('kW')
        plt.ylabel('Density')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"results/hourly_mean_dist_{level}_{input_dim}.png")
        plt.close()

        # Boxplot: Hourly Peak Distributions
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        data = [all_gen_stats[loss][level]['hourly_peak'].flatten() for loss in loss_names]
        plt.boxplot(data + [all_real_stats[loss_names[0]][level]['hourly_peak'].flatten()], labels=loss_names + ['Real'])
        plt.title(f'Half Hourly Peak Distribution - {level}, {input_dim}')
        plt.ylabel('kW')
        plt.xticks(rotation=45)
        plt.grid()

        plt.subplot(2, 1, 2)

        for i, d in enumerate(data + [all_real_stats[loss_names[0]][level]['hourly_peak'].flatten()]):
            sns.kdeplot(d, label=labels[i], linewidth=2)


        #plt.title(f'Daily Peak Distribution - {level} (Density)')
        plt.xlabel('kW')
        plt.ylabel('Density')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig(f"results/hourly_peak_dist_{level}_{input_dim}.png")
        plt.close()

def plot_hourly_error_boxplots(power_values, input_dim):
   
    
    loss_names = list(power_values.keys())
    half_hours = np.arange(24)
    levels = ['aggregate', 'household']

    n_rows, n_cols = len(loss_names), len(levels)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(7 * n_cols, 4 * n_rows),
                              sharex=True, sharey=False)

    # Normalize axes to a consistent 2D array regardless of n_rows/n_cols
    axes = np.atleast_2d(axes)
    if n_rows == 1 and n_cols > 1:
        axes = axes.reshape(1, n_cols)
    elif n_cols == 1 and n_rows > 1:
        axes = axes.reshape(n_rows, 1)

    for r, loss in enumerate(loss_names):
        for c, level in enumerate(levels):
            ax = axes[r, c]

            real_arr = power_values[loss]['real']
            gen_arr  = power_values[loss]['synth']

            if level == "aggregate":
                # Sum over households -> (48, C)
                real_stat = real_arr.sum(axis=2, keepdims=False).reshape(real_arr.shape[0], 24, 2).mean(axis=2)
                gen_stat  = gen_arr.sum(axis=2, keepdims=False).reshape(gen_arr.shape[0], 24, 2).mean(axis=2)
            else:  # household
                # Keep per-household resolution: flatten (days, households) per half-hour
                real_stat = real_arr.reshape(real_arr.shape[0], -1) \
                    if real_arr.shape[0] == 24 else real_arr.reshape(-1, real_arr.shape[-1]).T
                gen_stat = gen_arr.reshape(gen_arr.shape[0], -1) \
                    if gen_arr.shape[0] == 24 else gen_arr.reshape(-1, gen_arr.shape[-1]).T

            error = gen_stat - real_stat

            if error.ndim == 1:
                box_data = [[error[t]] for t in half_hours]
            else:
                if error.shape[0] != 24:
                    error = error.T   # -> (48, n)
                box_data = [error[t] for t in half_hours]

            ax.boxplot(box_data, positions=half_hours, widths=0.6,
                       patch_artist=True,
                       boxprops=dict(facecolor='steelblue', alpha=0.6),
                       medianprops=dict(color='red', linewidth=1.5),
                       flierprops=dict(marker='.', markersize=2, alpha=0.3),
                       whiskerprops=dict(linewidth=1),
                       capprops=dict(linewidth=1))
            #ax.axhline(0, color='black', linewidth=1.2, linestyle='--', label='Zero error')
            ax.set_title(f'{loss} — {level.capitalize()}', fontsize=16)
            if level == 'aggregate':
                ax.set_ylabel('Error (kW)', fontsize=18)
            ax.grid(axis='y', alpha=0.4)
            ax.tick_params(axis='y', labelsize=18)
            #ax.legend(fontsize=8)

    for c in range(n_cols):
        #axes[-1, c].set_xlabel('Half-Hour')
        axes[-1, c].set_xticks(half_hours[::2])
        axes[-1, c].set_xticklabels([f'{h:02d}:00' for h in half_hours[::2]],
                                      rotation=45, fontsize=18)
    

    plt.tight_layout()
    plt.savefig(f"results/hourly_error_boxplot_{input_dim}.png",
                dpi=150, bbox_inches='tight')
    plt.close()


#Causality plots

#Global  CATE plot
def plot_CATE_time(results_df, customer_size):

    plt.figure(figsize=(10, 7), dpi=300)

    # Group by mean solar penetration and calculate mean and quantiles for CATE
    mean_cate_temp = results_df.groupby('mean_t')['cate_linear'].agg(['mean']).reset_index()
    mean_cate_temp['lower_ci'] = results_df.groupby('mean_t')['cate_linear'].quantile(0.1).values
    mean_cate_temp['upper_ci'] = results_df.groupby('mean_t')['cate_linear'].quantile(0.9).values

    # Apply Gaussian smoothing
    #Since bootstap estimates can be noisy , smoothing reveals underlying trends 
    sigma_smoothing = 1.5 * np.std(mean_cate_temp['mean'])#larger sigma = more smooothing based on data variability
    mean_cate_temp['smoothed_mean'] = gaussian_filter1d(mean_cate_temp['mean'], sigma=sigma_smoothing)
    mean_cate_temp['smoothed_lower_ci'] = gaussian_filter1d(mean_cate_temp['lower_ci'], sigma=sigma_smoothing)
    mean_cate_temp['smoothed_upper_ci'] = gaussian_filter1d(mean_cate_temp['upper_ci'], sigma=sigma_smoothing)


    # Plotting the smoothed mean CATE against mean wind penetration with 80% CI
    fig, ax1 = plt.subplots(figsize=(6, 6), dpi=300)
    l1, = ax1.plot(mean_cate_temp['smoothed_mean'], mean_cate_temp['mean_t'],
                  linestyle='-', color="blue", label=f'Mean CATE', lw=2)

    ax1.fill_betweenx(mean_cate_temp['mean_t'], 
                     mean_cate_temp['smoothed_lower_ci'],
                     mean_cate_temp['smoothed_upper_ci'],
                     color="lightblue", alpha=1)
        
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.ylabel('Temperature (degrees)', fontsize=12)
    plt.xlabel('CATE (kW / degree)', fontsize=12)
    #plt.title('Temperature Sensitivity (CATE)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"results/causality/Winter_risk/Mean_CATE_Temp_PV_{customer_size}.png")
    plt.show()

    plt.scatter(results_df['cate_linear'], results_df['mean_t'], color="black", linestyle="dashed", label="True mean")

    plt.axvline(0, color='k', linestyle='-', lw=1, alpha=0.5)  # Add a vertical line at x=0
    plt.xlabel('CATE (kW/degree)', fontsize=16)
    plt.ylabel('Temperature (degrees)', fontsize=16)
    plt.grid(axis='x', linestyle='-', alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"results/causality/Winter_risk/Scatter_CATE_Temp_PV_{customer_size}.png")
    plt.show()

#Control split CATE plot
def plot_seasonal_hourly_cate(results_df, dataset, label):

    results_df['control'] = results_df['control'].astype(float)
    results_df['rv'] = results_df['rv'].astype(float)
    
    caps = sorted(results_df['control'].unique())

    results_df[['day_type', 'hour_str']] = results_df['label'].str.split('_', expand=True)

    unique_day = [d for d in results_df['day_type'].unique()]


    fig, axes = plt.subplots(1, len(unique_day), 
                             figsize=(8, 5),  constrained_layout=True,
                             sharey=True, dpi=300)
    ax_cate, ax_rv = axes[0], axes[1]

    # Define Color Map for Hours (0-23)
    cmap = plt.get_cmap('turbo')
    norm = mcolors.Normalize(vmin=min(caps) if len(caps)>1 else 0, 
                             vmax=max(caps) if len(caps)>1 else 10)

    
    for i, day in enumerate(unique_day):

        day_data = results_df[results_df['day_type'] == day]

        for c, colour in zip(caps, [cmap(norm(c)) for c in caps]):
            df_subset = day_data[day_data['control'] == c]
            #color = cmap(norm(c))

            grouped_cate = df_subset.groupby('mean_t')['cate_linear'].agg(['mean']).reset_index()
            grouped_cate['lower_ci'] = df_subset.groupby('mean_t')['cate_linear'].quantile(0.05).values
            grouped_cate['upper_ci'] = df_subset.groupby('mean_t')['cate_linear'].quantile(0.95).values
            grouped_cate = grouped_cate.sort_values('mean_t')

            #sigma_cate = max(0.5, 1.2 * np.std(grouped_cate['mean'])) if len(grouped_cate) > 1 else 1.0
            #grouped_cate['smooth_mean'] = gaussian_filter1d(grouped_cate['mean'], sigma=sigma_cate)
            #grouped_cate['smooth_low']  = gaussian_filter1d(grouped_cate['lower_ci'], sigma=sigma_cate)
            #grouped_cate['smooth_high'] = gaussian_filter1d(grouped_cate['upper_ci'], sigma=sigma_cate)

            # Plotting (Use smoothed_mean now)
            axes[i].plot(grouped_cate['mean'], grouped_cate['mean_t'], 
                         color=colour, alpha=0.8, linewidth=1.8, label=f'{c} kW')
            axes[i].fill_betweenx(grouped_cate['mean_t'], grouped_cate['lower_ci'], grouped_cate['upper_ci'], 
                                  color=colour, alpha=0.2)
            """
            grouped_rv = df_subset.groupby('mean_t')['rv'].agg(['mean']).reset_index()
            grouped_rv['lower_ci'] = df_subset.groupby('mean_t')['rv'].quantile(0.05).values
            grouped_rv['upper_ci'] = df_subset.groupby('mean_t')['rv'].quantile(0.95).values
            grouped_rv = grouped_rv.sort_values('mean_t')

            # Synchronized smoothing for the RV bounds
            #sigma_rv = max(0.5, 1.2 * np.std(grouped_rv['mean'])) if len(grouped_rv) > 1 else 1.0
            #grouped_rv['smooth_mean'] = gaussian_filter1d(grouped_rv['mean'], sigma=sigma_rv)
            #grouped_rv['smooth_low']  = gaussian_filter1d(grouped_rv['lower_ci'], sigma=sigma_rv)
            #grouped_rv['smooth_high'] = gaussian_filter1d(grouped_rv['upper_ci'], sigma=sigma_rv)

            axes[i].plot(grouped_rv['mean'], grouped_rv['mean_t'], 
                       color=colour, alpha=0.8, linewidth=1.8, linestyle='--')
            axes[i].fill_betweenx(grouped_rv['mean_t'], grouped_rv['lower_ci'], grouped_rv['upper_ci'], 
                                color=colour, alpha=0.2)
            """
        #axes[i, 1].axvline(0.1, color='black', linestyle=':', linewidth=1.5, alpha=0.8, 
                    #  label='Sensitivity Threshold ($\gamma = 0.1$)')
    

        # Formats and Label Management
        #ax_cate.set_title(f"Causal Temperature Sensitivity - Feeder {size}", fontsize=12, fontweight='bold', pad=10)
        axes[i].set_xlabel("CATE (kW/°C)", fontsize=14)
        axes[0].set_ylabel(f"Temperature (°C)", fontsize=14)
        axes[i].axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.4)
        axes[i].grid(True, alpha=0.25)
        axes[i].tick_params(axis='both', which='major', labelsize=14)
        axes[i].tick_params(axis='both', which='minor', labelsize=14)
        #ax_rv.set_title(f"Omitted Confounding Robustness Profile - Feeder {size}", fontsize=12, fontweight='bold', pad=10)
        #axes[i].set_xlabel("Robustness Value ($RV$)", fontsize=11)
        #axes[i].set_xlim(0, 1) # Boundaries for exact RV representation
        #axes[i].grid(True, alpha=0.25)
    
        # Combined cleanly formatted legends
        #ax_cate.legend(title="Hour Levels", fontsize=9, loc='upper center', framealpha=0.8)
        #ax_rv.legend(fontsize=9, loc='upper center', framealpha=0.8)      
        #plt.suptitle(f"Feeder {size}", fontsize=12, fontweight='bold')
        # Add Colorbar for Time of Day
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05, shrink =0.7)
    #plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(c)), markersize=10, label=f'Capacity {c} kW') for c in caps], title="Intalled HP capacities (kW)", bbox_to_anchor=(1.05, 1), loc='upper left')
    #cbar.set_label("Hour of Day (0 = Midnight, 12 = Noon, 23 = Night)", fontsize=9, loc='bottom')
    #cbar.set_ticks([0, 6, 12, 18, 24])
    cbar.set_ticklabels(['Midnight', '5 AM', '10 AM', '5 PM', '10 PM'], fontsize=12)
    
        #plt.suptitle("Temperature Sensitivity by Season & Hour", fontsize=18, y=1.02)
    
    # Save
    plt.savefig(f"results/causality/{dataset}/Capacity_CATE_Profiles_{label}.png", bbox_inches='tight')
    plt.close()
    """
    fig, axes = plt.subplots(1, len(unique_day), 
                             figsize=(6*len(unique_day), 10), 
                             sharey=True, dpi=300)
    #CI bar plot
    for i, day in enumerate(unique_day):

            day_data = results_df[results_df['day_type'] == day]
        #for j, season in enumerate(unique_seasons):

            #season_data = day_data[day_data['season'] == season]
        
            # Loop through Hours (Curves)
            hours = sorted(day_data['hour'].unique())
        
            for hour in hours:
                df_subset = day_data[day_data['hour'] == hour]

                grouped = df_subset.groupby('mean_temp')['cate_linear'].agg(['mean'])
                grouped['lower_ci'] = df_subset.groupby('mean_temp')['cate_linear'].quantile(0.05)
                grouped['upper_ci'] = df_subset.groupby('mean_temp')['cate_linear'].quantile(0.95)
                 

                grouped = grouped.reset_index().sort_values('mean_temp')


                # APPLY FILTERING
                sigma_smoothing = 1.5 * np.std(grouped['mean'])
                grouped['smoothed_mean'] = gaussian_filter1d(grouped['mean'], sigma=sigma_smoothing)
                grouped['smoothed_lower_ci'] = gaussian_filter1d(grouped['lower_ci'], sigma=sigma_smoothing)
                grouped['smoothed_upper_ci'] = gaussian_filter1d(grouped['upper_ci'], sigma=sigma_smoothing)
                grouped['width'] = grouped['smoothed_upper_ci'] - grouped['smoothed_lower_ci']
                color = cmap(norm(hour))
                print(grouped)

                #Box plot
                axes[i].plot(grouped['mean_temp'], grouped['width'], color=color)

            axes[i].set_xlabel("Mean Temperature (degrees)", fontsize=16)
            axes[i].grid(True, alpha=0.3)
        
            axes[i].set_ylabel("CI Width", fontsize=14)
    
    # Add Colorbar for Time of Day
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.08)
    cbar.set_label("Hour of Day (0 = Midnight, 12 = Noon, 23 = Night)", fontsize=16)
    cbar.set_ticks([0, 6, 12, 18, 24])
    cbar.set_ticklabels(['Midnight', '6 AM', 'Noon', '6 PM', 'Midnight'])
    
    plt.suptitle("95% CI of each Day Type and Hours Across Temperature", fontsize=18, y=1.02)
    
    # Save
    plt.savefig(f"results/causality/CI_CATE_Profiles{size}.png", bbox_inches='tight', dpi=300)
    plt.show()
    """
                

def visualize_CATE_temporal(df, models, size):
    labels =df['day_type'].astype(str) + '_' + df['hour'].astype(str)
    df['sensitivity'] = 0.0

    for label, group_indices in df.groupby(labels).groups.items():
        if label in models:
            # Get the correct physics curve for this bucket
            func = models[label]
            # Get the temperatures for these specific rows
            row_temps = df.loc[group_indices, 'temperature']
            # Calculate sensitivity
            df.loc[group_indices, 'sensitivity'] = func(row_temps)

    #Monthly CATE map
    matrix_month = (df.groupby(['month', 'hour'])['sensitivity'].mean().unstack())*(-1)
    
    # Create the X, Y Grid
    # X = Hours (Columns), Y = Months (Index)
    X_m, Y_m = np.meshgrid(matrix_month.columns, matrix_month.index)
    Z_m = matrix_month.values

    # Plot
    fig = plt.figure(figsize=(12, 12), dpi=300)
    ax1 = fig.add_subplot(111, projection='3d')

    # Plot Surface
    # rstride/cstride controls the 'step size' of the wireframe (smoothness)
    surf1 = ax1.plot_surface(X_m, Y_m, Z_m, cmap='RdYlBu_r', 
                             edgecolor='k', lw=0.1, alpha=0.9)

    # Labels & Formatting
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_title('Substation Risk Map (Month vs Hour)', fontsize=18, pad=20)
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('Month', fontsize=12)
    ax1.set_zlabel('CATE=', fontsize=12)

    ax1.set_yticks(range(12))
    ax1.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Adjust viewing angle (Elevation, Azimuth)
    ax1.view_init(elev=30, azim=-60)
    
    # Add Colorbar
    cbar = fig.colorbar(surf1, ax=ax1, shrink=0.3, aspect=10)
    cbar.ax.tick_params(labelsize=12) 

    cbar.set_label('CATE', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"results/causality/{dataset}/CATE_month_hour_{size}.png")
    plt.close()

    #Day of Week vs Hour (Weekly Fingerprint)
    # Create the matrix
    matrix_dow = df.groupby(['day', 'hour'])['sensitivity'].mean().unstack()*(-1)
    print(matrix_dow)
    
    # Create Grid
    X_d, Y_d = np.meshgrid(matrix_dow.columns, matrix_dow.index)
    Z_d = matrix_dow.values

    # Plot
    fig = plt.figure(figsize=(12, 12), dpi=300)
    ax2 = fig.add_subplot(111, projection='3d')

    surf2 = ax2.plot_surface(X_d, Y_d, Z_d, cmap='RdYlBu_r', 
                             edgecolor='k', lw=0.1, alpha=0.9)

    # Custom Y-Axis Labels for Days
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_title(f'Substation Weekly {label} Effect ', fontsize=18, pad=20)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Day of Week')
    ax2.set_zlabel('CATE ')
    
    # Set Y-ticks to be Mon-Sun instead of 0-6
    ax2.set_yticks(range(7))
    ax2.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    ax2.view_init(elev=35, azim=-45) # Different angle for better view
    
    cbar = fig.colorbar(surf2, ax=ax2, shrink=0.3, aspect=12)
    cbar.ax.tick_params(labelsize=12) 

    cbar.set_label('CATE=', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"results/causality/{dataset}/CATE_dow_hour_{size}.png")
    plt.close()

#Only for global CATE
def plot_CATE_with_sensitivity_temp(results_df, dataset, feeder):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Panel 1: CATE estimate vs true
    axes[0].plot(results_df['cate_linear'], results_df['mean_t'],
                 color='blue', label='Estimated CATE')
    axes[0].set_xlabel('CATE (kW/°C)')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Estimated vs True CATE')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Robustness Value per window
    axes[1].plot(results_df['rv'], results_df['mean_t'],
                 color='darkgreen', label='RV (q=1)')
    axes[1].plot(results_df['rva'], results_df['mean_t'],
                 color='limegreen', linestyle='--', label='RV (α=0.05)')
    #axes[1].axvline(0.1, color='orange', linestyle='--', label='RV=0.1 (fragile)')
    axes[1].set_xlabel('Robustness Value')
    axes[1].set_title('Sensitivity: Robustness Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Partial R² of treatment per window
    axes[2].plot(results_df['partial_r2'], results_df['mean_t'],
                 color='purple', label='Partial R² of t_res')
    axes[2].set_xlabel('Partial R² of Treatment')
    axes[2].set_title('Treatment Explanatory Power')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('CATE and Sensitivity by Temperature Window', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"results/causality/{dataset}/CATE_temp_{feeder}.png")
    plt.close()

def plot_CATE_with_sensitivity(results_df, dataset, feeder, label):
    # Create the figure with an extra column for the colorbar
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    
    # Define common mapping parameters
    cmap = 'RdYlBu_r' # Red for warm, Blue for cold
    temp_values = results_df['mean_t']
    capacity = results_df['control']

    mean_cate_temp = results_df.groupby('mean_t')['cate_linear'].agg(['mean']).reset_index()
    mean_cate_temp['lower_ci'] = results_df.groupby('mean_t')['cate_linear'].quantile(0.1).values
    mean_cate_temp['upper_ci'] = results_df.groupby('mean_t')['cate_linear'].quantile(0.9).values
    mean_cate_temp['control'] = results_df.groupby('mean_t')['control'].mean().values # Round for cleaner x-axis labels

    mean_rv_temp = results_df.groupby('mean_t')['rv'].agg(['mean']).reset_index()
    mean_rv_temp['rva'] = results_df.groupby('mean_t')['rva'].mean().values
    mean_rv_temp['lower_ci_rv'] = results_df.groupby('mean_t')['rv'].quantile(0.1).values
    mean_rv_temp['upper_ci_rv'] = results_df.groupby('mean_t')['rv'].quantile(0.9).values
    mean_rv_temp['control'] = results_df.groupby('mean_t')['control'].mean().values # Round for cleaner x-axis labels

    mean_r2_temp = results_df.groupby('mean_t')['r2yu_tw'].agg(['mean']).reset_index()
    mean_r2_temp['lower_ci_r2'] = results_df.groupby('mean_t')['r2yu_tw'].quantile(0.1).values
    mean_r2_temp['upper_ci_r2'] = results_df.groupby('mean_t')['r2yu_tw'].quantile(0.9).values
    mean_r2_temp['control'] = results_df.groupby('mean_t')['control'].mean().values # Round for cleaner x-axis labels


  

    # Panel 1: CATE estimate
    sc1 = axes[0].scatter(mean_cate_temp['mean'], mean_cate_temp['control'],
                          c=mean_cate_temp['mean_t'], cmap=cmap, edgecolors='k', alpha=0.7, label='Est. CATE')


    #add_trendline(axes[0], results_df['effect_min'], capacity_10, 'blue')
    axes[0].set_xlabel('CATE (kW/°C)')
    axes[0].set_ylabel('Hour of day')
    axes[0].set_title('Estimated CATE')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Robustness Value (RV)
    sc2 = axes[1].scatter(mean_rv_temp['mean'], mean_cate_temp['control'],
                          c=mean_rv_temp['mean_t'], cmap=cmap, edgecolors='k', alpha=0.7, label='RV (q=1)')
    # Also adding the alpha-level RV as smaller dots to keep visual focus
    axes[1].scatter(mean_rv_temp['rva'], mean_cate_temp['control'],
                    c=mean_rv_temp['mean_t'], cmap=cmap, marker='x', s=20, alpha=0.5, label='RV (α=0.05)')

    #axes[1].axvline(0.1, color='orange', linestyle=':', label='RV=0.1 (fragile)')
    axes[1].set_xlabel('Robustness Value')
    axes[1].set_title('Sensitivity: Robustness Value')
    axes[1].legend(loc='lower right', fontsize='small')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Partial R²
    sc3 = axes[2].scatter(mean_r2_temp['mean'], mean_r2_temp['control'],
                          c=mean_r2_temp['mean_t'], cmap=cmap, edgecolors='k', alpha=0.7)

    axes[2].set_xlabel('Partial R² of Treatment')
    axes[2].set_title('Treatment Explanatory Power')
    axes[2].grid(True, alpha=0.3)

    # Add Colorbar
    plt.subplots_adjust(right=0.9) # Make room for colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(sc1, cax=cbar_ax)
    cbar.set_label(f'Mean {label}', rotation=270, labelpad=15)

    plt.suptitle('CATE and Sensitivity Analysis (Color = Mean Temp Window)', fontsize=14, y=0.98)
    plt.savefig(f"results/causality/{dataset}/CATE_{feeder}_{label}.png")
    plt.close()

def plot_risk_distribution(risk_df,
                           label_feeders={14, 1, 71, 151},
                           high_risk={71, 151},
                           attenuating={1},
                           low_risk={14},
                           rv_threshold=0.1,  # Passed explicitly as a parameter
                           output_path="results/causality/risk_worst_case_v2.png"):

    fig, ax = plt.subplots(figsize=(10, 4.5))
 
    # --- Colour assignment ---
    def get_color(fid):
        if fid in high_risk:   return '#8B0000'
        if fid in attenuating: return '#E07B39'
        if fid in low_risk:    return '#2E8B57'
        return '#AAAAAA'

    
    # Scale marker size by capacity range safely
    feeder_range = risk_df['feeder_size'].clip(lower=1)
    sizes = 10 + (feeder_range * 5)

    #Uptake 
    color_metric = risk_df['cap_range'] 
    cmap = plt.cm.get_cmap('turbo')
 
    # Core Scatter Plot
    scatter = ax.scatter(
        risk_df['max_capacity'],
        risk_df['worst_case_cate'],
        c=color_metric,
        cmap=cmap,
        s=sizes,
        alpha=0.75,
        edgecolors='white',
        linewidths=0.4,
        zorder=3
    )

    # Add the continuous colorbar tracking absolute capacity intervals
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Heat Pump Uptake Range (kW)', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    errors = np.abs( risk_df['worst_case_cate'] - risk_df['cold_cate_p10'])
    asymmetric_error = [errors, np.zeros_like(errors)]

    ax.errorbar(risk_df['max_capacity'], risk_df['worst_case_cate'], yerr=asymmetric_error,  ecolor='grey', ls='none', capsize=3)
 
    # --- Reference lines & shading ---
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    

 
    # --- Annotate key feeders with dynamic text scaling ---
    # Calculates vertical offset relative to whole data distribution height
    #y_uptake = risk_df['worst_case_cate'].max() - risk_df['worst_case_cate'].min()
    #Calculate feeder size range for scaling
    y_span = risk_df['feeder_size'].max()
    dy_offset = (y_span * 0.002) if y_span > 0 else 0.001

    for _, row in risk_df[risk_df['feeder'].isin(label_feeders)].iterrows():
        fid = int(row['feeder'])
        x, y = row['max_capacity'], row['worst_case_cate']
 
        ax.annotate(
            f'Feeder {fid}',
            xy=(x, y),
            xytext=(x - 6, y + dy_offset),
            fontsize=9,
            fontweight='bold',
            color=get_color(fid),
            arrowprops=dict(arrowstyle='->', color=get_color(fid),
                            lw=1.2, connectionstyle='arc3,rad=0.15'),
            zorder=5
        )
 
 
    # --- Custom Legend Configuration ---
    # Standardizing fake elements to prevent layout crashes
    size_dummy_small = plt.Line2D([0], [0], marker='o', color='w', label='Small feeder size',
                                  markerfacecolor='grey', markersize=6, alpha=0.6)
    size_dummy_large = plt.Line2D([0], [0], marker='o', color='w', label='Large feeder size',
                                  markerfacecolor='grey', markersize=14, alpha=0.6)
    p10_dummy_CATE = plt.Line2D([0], [0], marker='_', label='Worst Case CATE Value',
                                  color='grey', markersize=10, alpha=0.6)

    legend_patches = [
        mpatches.Patch(color='#8B0000', label='High risk (medium/high capacity, high effect)'),
        mpatches.Patch(color='#E07B39', label='Medium risk (low capacity, high effect)'),
        mpatches.Patch(color='#2E8B57', label='Low Risk (high uptake, low effect)'),
        size_dummy_small,
        size_dummy_large,
        p10_dummy_CATE

    ]


    ax.legend(handles=legend_patches, loc='upper right', fontsize=8,
              framealpha=0.9,)
    # --- Labels and formatting ---
    ax.set_xlabel('HP Capacity Installed (kW)', fontsize=10)
    ax.set_ylabel('Worst-Case CATE at 10th Percentile (kW/°C)', fontsize=10)
    ax.set_title(
        'Operational Risk: Temperature Sensitivity at Peak HP Uptake',
        fontsize=12, fontweight='bold'
    )
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=10)
 
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    except FileNotFoundError:
        print("Output directory missing; displaying inline instead.")

 

def plot_capacity_distribution(feeder_dict):
    feeder_dict = {key: feeder_dict[key] for key in sorted(feeder_dict)}
    feeder_ids = list(feeder_dict.keys())
    halfway = (len(feeder_ids) + 1) // 2
    
    # Split feeders into two distinct groups
    groups = [feeder_ids[:halfway], feeder_ids[halfway:]]
    colors = ['#c6dbef', '#6baed6', '#2171b5']

    fig, axes = plt.subplots(2, 1, figsize=(24, 15), sharey=True)

    for g_idx, feeder_subgroup in enumerate(groups):
        ax = axes[g_idx]
    
        feeder_tick_positions, tick_labels = [], []

        for f_idx, fid in enumerate(feeder_subgroup):
            data = feeder_dict[fid]
            # Extract effects
            effect_data = data['values'].apply(np.array)
            effects = np.atleast_1d(effect_data).astype(float)
    
            # Sort by capacity size
            caps = np.atleast_1d(data['capacity'].astype(float))
            order = np.argsort(caps)
            caps = caps[order]
            
            feeder_center = f_idx * 3.0  # Spacing out feeders on the X-axis
            feeder_tick_positions.append(feeder_center)
            tick_labels.append(f'{fid}')
            
            x_pos = feeder_center 
                
            # Normalize effect values by the capacity size 
            ratios = effects 
               
            bp = ax.boxplot(ratios, positions=[x_pos], widths=0.5, 
                            patch_artist=True, showfliers=True)
                
            # Style the box color
            box_color = colors[0]
            for patch in bp['boxes']:
                patch.set_facecolor(box_color)
                patch.set_edgecolor('#4d4d4d')
                    
            # Style the whiskers, caps, and medians
            for element in ['whiskers', 'caps']:
                plt.setp(bp[element], color='#4d4d4d', linewidth=1.5)
            plt.setp(bp[element], color='#4d4d4d', linewidth=1.5)
            plt.setp(bp['medians'], color='crimson', linewidth=2)
                
            # Highlight the Minimum Value (Most Negative)
            min_ratio = np.min(ratios)
            ax.plot(x_pos, min_ratio, 'v', color='darkred', markersize=8, label='Min Value' if (f_idx==0) else "")
                
            # Indicating the capacity size below the marker cleanly without overlap
            # Changed va to 'top' and increased the offset slightly to dodge the triangle marker
            ax.text(x_pos, min_ratio - 0.12, f'{np.max(caps):.1f}',
                    ha='center', va='top', fontsize=10, color='#1a1a1a')
    
        ax.set_xticks(feeder_tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=12, rotation=45, ha='right')

        # Fixes the left-hand clipping
        first_x = 0
        last_x = (len(feeder_subgroup) - 1) * 3.0

        # Add a padding of 2.0 units to the left and right 
        # (Since your box widths are 0.5 and spacing is 3.0, 2.0 gives perfect breathing room)
        ax.set_xlim(first_x - 2.0, last_x + 2.0)

        ymin, ymax = ax.get_ylim()
        new_ymin = ymin - 0.3  # Expanded slightly to account for the capacity labels
        new_ymax = ymax + 0.05  
        ax.set_ylim(new_ymin, new_ymax)

        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_ylabel('Effect per kW installed (CATE / kW)', fontsize=12)
        
    plt.xlabel('Feeder ID', fontsize=14)
    plt.suptitle('Temperature Effect Distributions with Highlighted Maximum Capacity Installed', fontsize=14, fontweight='bold')
    
    # Using subplots_adjust instead of tight_layout gives better control over multi-row labels
    plt.subplots_adjust(hspace=0.3, bottom=0.1)
    
    plt.savefig(f"results/causality/CATE_cap_effects_feeders.png", bbox_inches='tight')

def plot_robustness_distributions(df, dataset, cate_col='cate_linear', rv_col='rv', r2_col='partial_r2'):
    output_path=f"results/causality/{dataset}/robustness_distribution.png"
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    row_specs = [
        (cate_col, 'CATE'),
        (rv_col, 'Robustness Value (rv)'),
        (r2_col, 'Partial R²'),
    ]
 
    for row_idx, (col, label) in enumerate(row_specs):
        ax_feat = axes[row_idx, 0]
        sns.boxplot(data=df, x='feature', y=col, ax=ax_feat)
        ax_feat.set_title(f'{label} by Feature')
        ax_feat.set_xlabel('Feature' if row_idx == 2 else '')
        ax_feat.set_ylabel(label)
        ax_feat.tick_params(axis='x', rotation=45)
 
        ax_hour = axes[row_idx, 1]
        sns.boxplot(data=df, x='hour', y=col, ax=ax_hour)
        ax_hour.set_title(f'{label} by Hour')
        ax_hour.set_xlabel('Hour of Day' if row_idx == 2 else '')
        ax_hour.set_ylabel(label)
 
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
 
