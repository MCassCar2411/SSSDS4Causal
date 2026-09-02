import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmark.wgan import Generator_linear, build_gan_scenarios
from diffusion import *
from DML.causality import calculate_sub_impact
from metrics import *
from network import *
from process_dataset import *
from visualisations import *


def set_seed(seed=42):
    random.seed(seed)
    # NumPy operations
    np.random.seed(seed)

    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_samples(
    file_path_train,
    config,
    csv_path,
    inference_data_path,
    n_samples,
    dataset,
    selected_date=None,
):
    train_dataset = build_diffusion_loaders(file_path_train, config, split="train")

    diffusion_model = Diffusion(config)
    device = config.train["device"]

    # Load trained model
    save_dir = f"train/SSSDS4/causal-diffusion/{dataset}/"  ##Update to LCL
    best_epoch = config.eval["epoch_id"]
    load_path = f"{save_dir}/epoch_{best_epoch}.pt"  # change to best epoch

    print(f"Loading weights from: {load_path}")
    state_dict = torch.load(load_path, map_location=device)

    for param in diffusion_model.denoiser.parameters():
        param.data = param.data.contiguous()

    for buf in diffusion_model.denoiser.buffers():
        buf.data = buf.data.contiguous()

    diffusion_model.denoiser.load_state_dict(state_dict)
    diffusion_model.denoiser.to(device)

    n_customers = config.train["batch_size"]
    start_date = "02/05/2023"
    end_date = "02/12/2023"
    label = "Winter"
    selected_date = pd.date_range(
        start=start_date, end=end_date, freq="30min"
    ).date.tolist()
    print(
        f"Generating {n_samples} synthetic samples for {n_customers} customers on {np.unique(selected_date)}"
    )

    new_dataset = build_diffusion_loaders(
        csv_path,
        config,
        split="validation",
        train_data=train_dataset.dataset,
        selected_date=selected_date,
    )  # Selected date?
    num_customers = config.train["batch_size"]
    samples = []
    date = pd.date_range(start=start_date, end=end_date, freq="30min").tolist()
    num_days = len(date) // 48
    calculate_sub_impact(
        csv_path, dataset=dataset, task="infer", selected_date=selected_date
    )
    for n in range(n_samples):
        # Inference
        generated_data = diffusion_model.infer(
            diffusion_model.denoiser,
            new_dataset,
            train_dataset,
            fixed_noise=False,
            type="Guide",
        )
        generated_dataset = reverse_load(generated_data, train_dataset)

        generated_dataset = generated_dataset.reshape(num_days, num_customers, 48, 1)

        generated_dataset = generated_dataset.transpose(0, 2, 1, 3).squeeze(
            -1
        )  # B, T, C
        generated_dataset = generated_dataset.reshape(-1, num_customers)

        samples.append(generated_dataset)

    samples = np.array(samples)  # sample, T, C

    sample_aggregates = samples.sum(axis=2)
    mean_data_final = samples.mean(axis=0)

    # Ensure T matches the generated data length
    T = mean_data_final.shape[0]
    time_index = date[:T]

    # Save Individual Customer CSVs (Averaged over n_samples)
    output_folder = inference_data_path

    os.makedirs(output_folder, exist_ok=True)

    for i in range(num_customers):
        cust_df = pd.DataFrame(
            {"Timestamp": time_index, "Load_kW": mean_data_final[:, i]}
        )
        cust_df.to_csv(
            os.path.join(output_folder, f"Customer_{i + 1}.csv"), index=False
        )

    # Save the Aggregate CSV (Sum of the averaged customers)
    aggregate_load = mean_data_final.sum(axis=1)
    df_agg_save = pd.DataFrame(
        {"Timestamp": time_index, "Aggregate_Load_kW": aggregate_load}
    )
    df_agg_save.to_csv(
        os.path.join(inference_data_path, "Aggregate_Synthetic.csv"), index=False
    )

    # We need the individual sample aggregates to show the "grey lines" of uncertainty
    sample_aggregates = samples.sum(axis=2)  # Shape: (n_samples, Time)
    x0_days = []

    for batch in new_dataset:
        x0 = batch["agg_data"][0][0].cpu().numpy()
        print(x0)
        x0_days.append(x0)

    # Execute Plot
    plot_aug_synthetic_data(
        aggregate=sample_aggregates,
        ind=samples,
        x0=np.array(x0_days),
        n_samples=n_samples,
        input_dim=num_customers,
        label=label,
        mean_aggregate=aggregate_load,
    )


def train_diffusion(config, train_loader, dataset):
    diffmodel = Diffusion(config)
    model = diffmodel.denoiser.to(config.train["device"])
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    train_config = config.train

    save_dir = f"train/SSSDS4/causal-diffusion/{dataset}/"

    os.makedirs(save_dir, exist_ok=True)

    criterion_mse = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=train_config["lr"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_config["num_epochs"]
    )

    train_losses = []
    train_time = []
    model.train()

    for epoch in range(train_config["num_epochs"]):
        total_train_loss = 0
        start_time = time.time()

        for batch in train_loader:
            optimizer.zero_grad()
            x0 = batch["target_data"].to(config.train["device"])  # (B, K, L)
            conditions = batch["cond_data"].to(
                config.train["device"]
            )  # (B, num_exogenous, L)

            B, _, _ = x0.shape

            t = torch.randint(
                0, config.diffusion["T"], (B,), device=config.train["device"]
            )
            # Forward Diffusion
            noisy_data, noise = diffmodel.forward_diffusion(x0, t)
            # Predict noise added
            predicted = model(noisy_data, conditions, t)  # (B,K,L)
            # MSE
            train_loss = criterion_mse(predicted, noise)
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()

        end_time = time.time()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_time.append(end_time - start_time)

        save_path = f"{save_dir}/epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), save_path)

        print(
            f" Epoch {epoch + 1}/{train_config['num_epochs']} - Train Loss: {avg_train_loss:.4f}, Training Time: {train_time[-1]}s"
        )
        lr_scheduler.step()
    plot_training_curve(train_losses, "mse", config.num_targets)
    print(f" Average training time: {np.mean(train_time)}s")


def evaluate(
    config,
    file_path_train,
    file_path_val,
    dataset="elektro",
    seed=42,
    test_pvalue=False,
):

    set_global_seed(seed)
    if test_pvalue:
        evaluate_p(
            config,
            file_path_train,
            file_path_val,
            config.train["device"],
            dataset=dataset,
        )

    models = {"WGAN", "SSSDS4+causal", "SSSDS4"}  # , "CREST", "elexon"}
    # models = {"SSSDS4+causal", "SSSDS4"}
    (
        admd_stats,
        power_values,
        other_scores,
        power_metrics,
        all_real_stats,
        all_gen_stats,
        aggregate_feat,
    ) = evaluate_models(
        config,
        file_path_val,
        file_path_train,
        config.input_dim,
        config.train["device"],
        models,
        best_epoch=config.eval["epoch_id"],
        seed=seed,
        dataset=dataset,
    )

    label = f"{dataset}"
    plot_mean_data(power_values, label=f"{dataset}")
    plot_power_stats_by_loss(all_real_stats, all_gen_stats, power_metrics, label)
    plot_hourly_error_boxplots(power_values, label)

    # Plot ADMD/MDMD and sample distributions
    plot_all_testing_results(other_scores, admd_stats, aggregate_feat, label)


def evaluate_models(
    config,
    file_path_val,
    file_path_train,
    input_dim,
    device,
    losses={"mse"},
    n_samples=500,
    best_epoch=200,
    seed=42,
    dataset="elektro",
):

    all_admd_stats = {}
    all_scores = {}
    all_real_stats = {}
    all_gen_stats = {}
    power_metrics = {}
    power_values = {}
    agg_features = {}

    real_data_list, x_t_val_list, agg_data_list = [], [], []

    train_loader = build_diffusion_loaders(file_path_train, config, split="train")
    val_loader = build_diffusion_loaders(
        file_path_val, config, split="validation", train_data=train_loader.dataset
    )

    # Extract val data and corresponding aggregate data, and sample x_t for inference (same for losses to ensure fair comparison)
    for batch in val_loader:
        x0 = batch["target_data"].to(device)  # Individaul
        agg_real = batch["agg_data"][0][0].to(device)  # Real aggregate
        real_data_list.append(x0.detach().cpu().numpy())

        agg_data_list.append(agg_real.detach().cpu().numpy())

        torch.manual_seed(seed)  # Before sampling x_t
        x_t = torch.randn(
            (x0.shape[0], x0.shape[1], x0.shape[2]), device=device
        )  # Start from pure noise
        x_t_val_list.append(x_t)

    x_t_data = torch.cat(tensors=x_t_val_list, axis=0)  # type: ignore

    torch.save(x_t_data, "train/x_t.pt")
    real_dataset = np.concatenate(real_data_list, axis=0)
    size = config.train["batch_size"]
    B, C, T = real_dataset.shape
    num_days = B // size
    print(num_days)

    agg_real = np.concatenate(agg_data_list, axis=0).reshape(num_days, T)
    print(real_dataset.shape)
    real_dataset = real_dataset.reshape(num_days, size, T, 1)
    real_dataset = real_dataset.transpose(0, 2, 1, 3).squeeze(-1)  # B, T, C

    for loss_name in losses:
        print(loss_name)
        if loss_name == "SSSDS4+causal" or "SSSDS4":
            print(config)
            diffusion_model = Diffusion(config)  ##
            config.eval["epoch_id"] = 200
            best_epoch = config.eval["epoch_id"]
            save_dir = f"train/SSSDS4/causal-diffusion/{dataset}"
            load_path = f"{save_dir}/epoch_{best_epoch}.pt"  # change to best epoch

            print(f"Loading weights from: {load_path}")

            state_dict = torch.load(load_path, map_location=device)

            for param in diffusion_model.denoiser.parameters():
                param.data = param.data.contiguous()

            for buf in diffusion_model.denoiser.buffers():
                buf.data = buf.data.contiguous()

            diffusion_model.denoiser.load_state_dict(state_dict)
            diffusion_model.denoiser.to(device)
            diffusion_model.denoiser.eval()

            for param in diffusion_model.denoiser.parameters():
                param.requires_grad = False
            if loss_name == "SSSDS4":
                with torch.no_grad():
                    generated_dataset = pd.read_csv(
                        save_dir + "_Diff_synth.csv", index_col=0
                    )
                    print(generated_dataset.shape)
                    generated_dataset = generated_dataset.values.reshape(
                        num_days, T, size
                    )

            elif loss_name == "SSSDS4+causal":
                with torch.set_grad_enabled(True):
                    generated_dataset = pd.read_csv(
                        save_dir + "_Diff_Causal_synth.csv", index_col=0
                    )

                    generated_dataset = generated_dataset.values.reshape(
                        num_days, T, size
                    )
                    print(generated_dataset.shape)
        if loss_name == "WGAN":
            load_path = f"train/WGANGP/load_WGANGP_1_0_{dataset}"

            max_cap = 1
            # Scenarios are generated into a dict of length nb days (#VS or # TEST sizes)
            # Each day of the dict is an array of shape (n_scenarios, 24)
            train_dataset_diff = build_diffusion_loaders(
                file_path_train, config, normalisation="minmax"
            )
            val_dataset_diff = build_diffusion_loaders(
                file_path_val,
                config,
                split="validation",
                train_data=train_dataset_diff.dataset,
            )

            indices = []
            non_null_indexes = list(
                np.delete(np.asarray([i for i in range(24)]), indices)
            )
            config_GAN = config.GAN_wasserstein
            # Instance generator neural network
            gen = Generator_linear(
                latent_s=config_GAN["latent_s"],
                cond_in=config_GAN["cond_in"],
                in_size=config_GAN["in_size"],
                gen_w=config_GAN["gen_w"],
                gen_l=config_GAN["gen_l"],
                gpu=device,
            )
            gen.weights_initialize(mean=0.0, std=0.02)
            gen.eval()
            gen = torch.load(load_path, weights_only=False)

            print(f"Loading weights from: {load_path}")

            # Sample MODEL
            generated_data = build_gan_scenarios(
                val_loader=val_dataset_diff,
                gen=gen,
                max=max_cap,
                gpu=device,
                tag="load",
                non_null_indexes=non_null_indexes,
            )

            print(generated_data.shape)

            generated_dataset = reverse_load(
                generated_data.reshape(num_days, T, size).transpose(0, 2, 1),
                train_dataset_diff,
            )  # C*T, B
            print(generated_dataset.shape)
            # split into customers and days
            # generated_dataset = generated_dataset.reshape(num_days, 158, T, 1)
            generated_dataset = generated_dataset.transpose(0, 2, 1)  # B, T, C
        if loss_name == "elexon":
            save_dir = f"train/Elexon/fval_{dataset}/"
            generated_data = pd.read_csv(save_dir + "Elexon_val.csv")

            generated_dataset = generated_data["Customer Loads"]
            generated_dataset = generated_dataset.apply(ast.literal_eval)
            generated_dataset = organise_customers(generated_dataset, T, size)

            generated_dataset = generated_dataset.reshape(num_days, size, T, 1)
            generated_dataset = generated_dataset.transpose(0, 2, 1, 3).squeeze(
                -1
            )  # B, T, C
        if loss_name == "CREST":
            save_dir = f"train/CREST/fval_{dataset}/"
            generated_data = pd.read_csv(save_dir + "CREST_val.csv")
            generated_dataset = generated_data["Customer_load"]
            generated_dataset = generated_dataset.apply(ast.literal_eval)
            generated_dataset = organise_customers(generated_dataset, T, size)
            print(generated_dataset.shape)

            generated_dataset = generated_dataset.reshape(num_days, size, T, 1)
            generated_dataset = generated_dataset.transpose(0, 2, 1, 3).squeeze(
                -1
            )  # B, T, C

            print(generated_dataset.shape)

        # ADMD
        admd_gen = compute_admd_mdmd(
            generated_dataset, label="synth", C=generated_dataset.shape[2]
        )
        admd_real = compute_admd_mdmd(agg_real, label="real", C=size)
        admd_score = wasserstein_distance(admd_real.flatten(), admd_gen.flatten())

        plot_peaks(generated_dataset, agg_real, label=loss_name + f"_{dataset}")
        plot_tsne(
            agg_real,
            generated_dataset,
            labels=("Real", "Synthetic"),
            label=loss_name + f"_{dataset}",
            level="agg",
        )

        avg_mmd_score_agg, _ = mmd(
            n_samples, agg_real, generated_dataset, level="agg", seeds=10, seed=seed
        )
        wd_score_agg, _ = wd(
            n_samples, agg_real, generated_dataset, level="agg", seeds=10, seed=seed
        )
        rmse_mae_agg = rmse_mae_score(
            agg_real,
            generated_dataset,
            level="agg",
            n_samples=n_samples,
            seeds=10,
            seed=seed,
        )
        rmse_agg = rmse_mae_agg["rmse_mean"]
        mae_agg = rmse_mae_agg["mae_mean"]

        print(
            f"{loss_name} - Epoch {best_epoch}: MMD = {avg_mmd_score_agg:.5f}, WD = {wd_score_agg:.5f}, RMSE = {rmse_agg:.4f}, MAE = {mae_agg:.4f}, ADMD = {admd_score:.5f}"
        )

        # Aggregate level
        print("These are real stats")
        real_stats_a = compute_power_stats(agg_real, label="real", level="aggregate")
        print("These are synth stats")
        gen_stats_a = compute_power_stats(
            generated_dataset, label="synth", level="aggregate"
        )

        dist_a = power_stat_distances(real_stats_a, gen_stats_a)
        print(dist_a)

        # Measure peak prominence and sharpness
        deriv_r = measure_peaks(agg_real, C, label=loss_name)
        deriv_s = measure_peaks(generated_dataset, C, label="synth")

        # Household level
        # Mask the missing customers in synth data
        for day in range(generated_dataset.shape[0]):
            for cust in range(generated_dataset.shape[1]):
                if np.sum(real_dataset[day, :, cust]) == 0:
                    generated_dataset[day, :, cust] = 0

        # Metrics
        avg_mmd_score_ind, _ = mmd(
            n_samples, real_dataset, generated_dataset, level="ind", seeds=10, seed=seed
        )
        wd_score_ind, _ = wd(
            n_samples, real_dataset, generated_dataset, level="ind", seeds=10, seed=seed
        )
        rmse_mae_ind = rmse_mae_score(
            real_dataset,
            generated_dataset,
            level="ind",
            n_samples=n_samples,
            seeds=10,
            seed=seed,
        )
        plot_tsne(
            real_dataset,
            generated_dataset,
            labels=("Real", "Synthetic"),
            label=loss_name + f"_{dataset}",
            level="ind",
        )
        rmse_ind = rmse_mae_ind["rmse_mean"]
        mae_ind = rmse_mae_ind["mae_mean"]
        print(
            f"{loss_name} - Epoch {best_epoch}: MMD = {avg_mmd_score_ind:.5f}, WD = {wd_score_ind:.5f}, RMSE = {rmse_ind:.4f}, MAE = {mae_ind:.4f}"
        )

        # Household level
        print("These are real stats")
        real_stats_h = compute_power_stats(
            real_dataset, label="real", level="household"
        )
        print("These are synth stats")
        gen_stats_h = compute_power_stats(
            generated_dataset, label="synth", level="household"
        )

        dist_h = power_stat_distances(real_stats_h, gen_stats_h)
        print(dist_h)

        agg_features[loss_name] = {"real": deriv_r, "synth": deriv_s}

        power_metrics[loss_name] = {"household": dist_h, "aggregate": dist_a}

        all_real_stats[loss_name] = {
            "household": real_stats_h,
            "aggregate": real_stats_a,
        }

        all_gen_stats[loss_name] = {"household": gen_stats_h, "aggregate": gen_stats_a}
        all_admd_stats[loss_name] = {"real": admd_real, "generated": admd_gen}

        all_scores[loss_name] = {
            "agg": {
                "mmd": avg_mmd_score_agg,
                "wd": wd_score_agg,
                "rmse": rmse_mae_agg["rmse_mean"],
                "mae": rmse_mae_agg["mae_mean"],
            },
            "ind": {
                "mmd": avg_mmd_score_ind,
                "wd": wd_score_ind,
                "rmse": rmse_mae_ind["rmse_mean"],
                "mae": rmse_mae_ind["mae_mean"],
            },
        }

        power_values[loss_name] = {"real": real_dataset, "synth": generated_dataset}

    return (
        all_admd_stats,
        power_values,
        all_scores,
        power_metrics,
        all_real_stats,
        all_gen_stats,
        agg_features,
    )


def evaluate_p(config, file_path_train, file_path_val, device, dataset="elektro"):
    n_samples = 500
    seed = 42
    real_data_list, x_t_val_list, agg_data_list = [], [], []

    train_loader = build_diffusion_loaders(file_path_train, config, split="train")
    val_loader = build_diffusion_loaders(
        file_path_val, config, split="validation", train_data=train_loader.dataset
    )

    # Extract val data and corresponding aggregate data, and sample x_t for inference (same for losses to ensure fair comparison)

    for batch in val_loader:
        x0 = batch["target_data"].to(device)  # Individaul
        agg_real = batch["agg_data"][0][0].to(device)  # Real aggregate
        real_data_list.append(x0.detach().cpu().numpy())

        agg_data_list.append(agg_real.detach().cpu().numpy())

        torch.manual_seed(seed)  # Before sampling x_t
        x_t = torch.randn(
            (x0.shape[0], x0.shape[1], x0.shape[2]), device=device
        )  # Start from pure noise
        x_t_val_list.append(x_t)

    x_t_data = torch.cat(x_t_val_list, axis=0)  # type: ignore

    torch.save(x_t_data, "train/x_t.pt")
    real_dataset = np.concatenate(real_data_list, axis=0)
    size = config.train["batch_size"]
    B, C, T = real_dataset.shape
    num_days = B // size
    print(num_days)

    agg_real = np.concatenate(agg_data_list, axis=0).reshape(num_days, T)
    print(real_dataset.shape)
    real_dataset = real_dataset.reshape(num_days, size, T, 1)
    real_dataset = real_dataset.transpose(0, 2, 1, 3).squeeze(-1)  # B, T, C

    diffusion_model = Diffusion(config)  ##
    config.eval["epoch_id"] = 200
    best_epoch = config.eval["epoch_id"]
    save_dir = f"train/SSSDS4/causal-diffusion/{dataset}"
    load_path = f"{save_dir}/epoch_{best_epoch}.pt"  # change to best epoch

    print(f"Loading weights from: {load_path}")

    state_dict = torch.load(load_path, map_location=device)

    for param in diffusion_model.denoiser.parameters():
        param.data = param.data.contiguous()

    for buf in diffusion_model.denoiser.buffers():
        buf.data = buf.data.contiguous()

    diffusion_model.denoiser.load_state_dict(state_dict)
    diffusion_model.denoiser.to(device)
    diffusion_model.denoiser.eval()

    for param in diffusion_model.denoiser.parameters():
        param.requires_grad = False

    p_dict = {}
    p_arr = [1, 10, 15, 20, 50, 60, 75, 100]  # Example p values to test
    with torch.set_grad_enabled(True):
        for p_val in p_arr:
            print(f"This is pval:{p_val}")
            if os.path.exists(
                os.path.join(os.getcwd(), save_dir, f"Diff_Causal_synth_{p_val}.csv")
            ):
                generated_dataset = pd.read_csv(
                    save_dir + f"/Diff_Causal_synth_{p_val}.csv", index_col=0
                )

                generated_dataset = generated_dataset.values.reshape(num_days, T, size)

            else:
                generated_data = diffusion_model.infer(
                    diffusion_model.denoiser,
                    val_loader,
                    train_loader,
                    fixed_noise=True,
                    type="Guide",
                    p=p_val,
                )  # replace by loading the same data
                generated_dataset = reverse_load(generated_data, train_loader)

                # split into customers and days
                generated_dataset = generated_dataset.reshape(num_days, size, T, 1)
                generated_dataset = generated_dataset.transpose(0, 2, 1, 3).squeeze(
                    -1
                )  # B, T, C
                pd.DataFrame(generated_dataset.reshape(-1, C)).to_csv(
                    save_dir + f"/Diff_Causal_synth_{p_val}.csv"
                )

            avg_mmd_score_agg, _ = mmd(
                n_samples, agg_real, generated_dataset, level="agg", seeds=10, seed=seed
            )
            wd_score_agg, _ = wd(
                n_samples, agg_real, generated_dataset, level="agg", seeds=10, seed=seed
            )
            # Household level
            # Mask the missing customers in synth data
            for day in range(generated_dataset.shape[0]):
                for cust in range(generated_dataset.shape[1]):
                    if np.sum(real_dataset[day, :, cust]) == 0:
                        generated_dataset[day, :, cust] = 0

            avg_mmd_score_ind, _ = mmd(
                n_samples,
                real_dataset,
                generated_dataset,
                level="ind",
                seeds=10,
                seed=seed,
            )
            wd_score_ind, _ = wd(
                n_samples,
                real_dataset,
                generated_dataset,
                level="ind",
                seeds=10,
                seed=seed,
            )

            p_dict[p_val] = {
                "agg": {"mmd": avg_mmd_score_agg, "wd": wd_score_agg},
                "ind": {"mmd": avg_mmd_score_ind, "wd": wd_score_ind},
            }

    plot_metric_line_chart(p_dict, dataset)
