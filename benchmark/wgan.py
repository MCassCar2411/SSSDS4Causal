"""Reimplement WGAN_GP Codebase.

Reference: Jonathan Dumas and Antoine Wehenkel and Damien Lanaspeze and Bertrand Cornelusse and Antonio Sutera,
"A deep generative model for probabilistic energy forecasting in power systems: normalizing flows"
Applied Energy, 2022.

Paper link: https://www.sciencedirect.com/science/article/pii/S0306261921011909

Code author: Jonathan Dumas

Updated by: Maria Cassidy (maria.cassidy-cararsco@strath.ac.uk)

Last updated Date: September 2nd 2026
"""

import os
import sys
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.cfg import Config
from process_dataset import *
from visualisations import *


def scale_data_multi(
    x_LS: np.array,
    y_LS: np.array,
    x_VS: np.array,
    y_VS: np.array,
    x_TEST: np.array,
    y_TEST: np.array,
):
    """
    Scale data for NFs multi-output.
    """
    y_LS_scaler = StandardScaler()
    y_LS_scaler.fit(y_LS)
    y_LS_scaled = y_LS_scaler.transform(y_LS)
    y_VS_scaled = y_LS_scaler.transform(y_VS)
    y_TEST_scaled = y_LS_scaler.transform(y_TEST)

    x_LS_scaler = StandardScaler()
    x_LS_scaler.fit(x_LS)
    x_LS_scaled = x_LS_scaler.transform(x_LS)
    x_VS_scaled = x_LS_scaler.transform(x_VS)
    x_TEST_scaled = x_LS_scaler.transform(x_TEST)

    return (
        x_LS_scaled,
        y_LS_scaled,
        x_VS_scaled,
        y_VS_scaled,
        x_TEST_scaled,
        y_TEST_scaled,
        y_LS_scaler,
    )


# Model Architecture
class Generator_linear(nn.Module):
    """
    Define Generator class.
    Conditional generator using fully connected layers.
    """

    def __init__(self, **kwargs):
        """
        Generator constructor
        :param latent_s: Dim of the latent space
        :param cond_in: Dim of context (weather forecasts, etc)
        :param hidden_layer: number of hidden layers
        :param neurons_per_layer : number of neurons per hidden_layer
        :param in_size: Dim of the random variable to model (PV, wind power, etc)
        """

        super().__init__()
        self.in_size = kwargs[
            "in_size"
        ]  # Dim of the random variable to model (PV, wind power, etc)
        self.cond_in = kwargs["cond_in"]  # Dim of context (weather forecasts, etc)
        self.latent_s = kwargs["latent_s"]  # Dim of the latent space

        # Set GPU if available
        if kwargs["gpu"]:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        l_gen_net = (
            [self.latent_s + self.cond_in]
            + [kwargs["gen_w"]] * kwargs["gen_l"]
            + [self.in_size]
        )

        # Build the generator
        self.gen_net = []
        for l1, l2 in zip(l_gen_net[:-1], l_gen_net[1:]):
            self.gen_net += [nn.Linear(l1, l2), nn.ReLU()]
        self.gen_net.pop()  # Regression problem, no activation function at the last layer
        self.gen = nn.Sequential(*self.gen_net)

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, noise: torch.Tensor, context: torch.Tensor):
        """
        Define forward pass
        :param noise: noise torch tensor
        :param context: conditional torch tensor
        :return: output torch tensor
        """

        pred = self.gen(torch.cat((noise, context), dim=1))

        return pred

    def weights_initialize(self, mean: float, std: float):
        """
        Initialize self model parameters following a normal distribution based on mean and std
        :param mean: mean of the standard distribution
        :param std : standard deviation of the normal distribution
        :return: None
        """
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, mean=mean, std=std)

    def sample(self, n_s=1, x_cond: np.array = None):
        """
        :param n_s: number of scenarios
        :param x_cond: context (weather forecasts, etc) into an array of shape (self.cond_in,)
        :return: samples into an array of shape (nb_samples, self.in_size)
        """
        # Generate samples from a multivariate Gaussian
        z = torch.randn(n_s, self.latent_s).to(self.device)
        context = (
            torch.tensor(np.tile(x_cond, n_s).reshape(n_s, self.cond_in))
            .to(self.device)
            .float()
        )
        scenarios = (
            self.gen(torch.cat((z, context), dim=1))
            .view(n_s, -1)
            .cpu()
            .detach()
            .numpy()
        )

        return scenarios


class Discriminator_wassertein(nn.Module):
    """
    Define critic class, discriminator using Wasserstein distance estimate.
    Return a positive number. Higher the output is, more realistic is the input.
    """

    def __init__(self, **kwargs):
        """
        Critic constructor
        :param input_dim: size of the input (real or fake) torch tensor
        :param condition_dim: size ot the conditional torch tensor
        :param hidden_layer: number of hidden layer
        :param neurons_per_layer : number of neurons per hidden layer
        """
        super().__init__()

        self.in_size = kwargs[
            "in_size"
        ]  # Dim of the random variable to model (PV, wind power, etc)
        self.cond_in = kwargs["cond_in"]  # Dim of context (weather forecasts, etc)
        self.latent_s = kwargs["latent_s"]  # Dim of the latent space
        self.lambda_gp = kwargs["lambda_gp"]

        # Set GPU if available
        if kwargs["gpu"]:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        l_dis_net = (
            [self.in_size + self.cond_in] + [kwargs["gen_w"]] * kwargs["gen_l"] + [1]
        )

        # Build the discriminator
        alpha = 0.01
        self.dis_net = []
        for l1, l2 in zip(l_dis_net[:-1], l_dis_net[1:]):
            self.dis_net += [nn.Linear(l1, l2), nn.LeakyReLU(alpha)]
        self.dis_net.pop()  # The last activation function is a ReLU to return a positive number
        self.dis_net.append(nn.ReLU())
        self.dis = nn.Sequential(*self.dis_net)

    def loss(
        self,
        generated_samples: torch.Tensor,
        true_samples: torch.Tensor,
        context: torch.Tensor,
    ):

        # Discriminator's answers to generated and true samples
        D_true = self.dis(torch.cat((true_samples, context), dim=1))
        D_generated = self.dis(torch.cat((generated_samples, context), dim=1))
        # Compute Discriminator's loss with a gradient penalty to force Lipschitz condition
        gp = self.grad_pen(
            real=true_samples, samples=generated_samples, context=context
        )
        loss = -(torch.mean(D_true) - torch.mean(D_generated)) + self.lambda_gp * gp

        return loss

    def forward(self, input: torch.Tensor, context: torch.Tensor):
        """
        Define forward pass
        :param input: input (real or fake) torch tensor
        :param context: conditional torch tensor
        :return: output torch tensor
        """

        pred = self.dis(torch.cat((input, context), dim=1))

        return pred

    def weights_initialize(self, mean: float, std: float):
        """
        Initialize self model parameters following a normal distribution based on mean and std
        :param mean: mean of the standard distribution
        :param std : standard deviation of the normal distribution
        :return: None
        """
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, mean=mean, std=std)

    def grad_pen(
        self, real: torch.Tensor, samples: torch.Tensor, context: torch.Tensor
    ):
        """
        For discriminator using Wasserstein distance estimate.
        Compute gradient penalty to add to critic loss in order to force Lipschitz condition using batch
        of interpolated sample. The Lipschitz constraint is obtained if the gradient is 1 on all interpolated
        sample. The gradient penalty forces this condition.
        :param real: batch of real samples, shape: batch_size * 24
        :param samples: batch of generated sample, shape: batch_size * 24
        :param context : batch of conditional torch tensor, shape: batch_size * 24
        :return: gradient penalty
        """

        # Interpolated sample
        bs, sample_size = real.shape[0], real.shape[1]
        epsilon = torch.rand((bs, sample_size), device=self.device)
        interpolated_sample = real * epsilon + samples * (1 - epsilon)
        # Compute critic scores
        mixed_score = self.dis(torch.cat((interpolated_sample, context), dim=1))
        # Gradient of the mixed_score with respect with the interpolated_sample
        gradient = torch.autograd.grad(
            inputs=interpolated_sample,
            outputs=mixed_score,
            grad_outputs=torch.ones_like(mixed_score),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_pen = torch.mean((gradient_norm - 1) ** 2)

        return gradient_pen


def fit_gan_wasserstein(
    nb_epoch: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    gen,
    dis,
    opt_gen,
    opt_dis,
    n_discriminator: int,
    batch_size: int = 100,
    wdb: bool = False,
    gpu: bool = True,
):
    """
    Fit GAN with discriminator using the Wasserstein distance estimate.
    """
    # to assign the data to GPU with .to(device) on the data
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    # Assign models and data to gpu
    gen.to(device)
    dis.to(device)
    real_data_list = []
    real_cond_list = []
    for batch in val_loader:
        y_VS_gpu = batch["target_data"].to(device)  # Individaul
        x_VS_gpu = batch["cond_data"].to(device)
        real_data_list.append(y_VS_gpu.detach().cpu().numpy())
        real_cond_list.append(x_VS_gpu.detach().cpu().numpy())

    real_dataset = np.concatenate(real_data_list, axis=0)
    real_cond = np.concatenate(real_cond_list, axis=0)

    B, _, _ = real_dataset.shape

    real_dataset = real_dataset.reshape(B, -1)  # B, C*T
    real_cond = real_cond.reshape(B, -1)  # B, C*T

    # x_VS_gpu = torch.tensor(x_VS).to(device).float()
    # y_VS_gpu = torch.tensor(y_VS).to(device).float()
    # x_TEST_gpu = torch.tensor(x_TEST).to(device).float()
    # y_TEST_gpu = torch.tensor(y_TEST).to(device).float()

    loss_list = []
    time_tot = 0.0

    # WARNING: batch size = 10 % #LS
    # batch_size = int(0.1 * y_LS.shape[0])

    for epoch in range(nb_epoch):
        start = timer()

        # Shuffle the data randomly at each epoch
        # seed = random.randint(0, 2000)
        # x_LS_shuffled, y_LS_shuffled = shuffle(x_LS, y_LS, random_state=seed)

        batch_dis_idx = 0
        batch_gen_idx = 0
        loss_D_batch = 0
        loss_G_batch = 0

        # Training batch loop
        # batch_list = [i for i in range(batch_size, batch_size * y_LS.shape[0] // batch_size, batch_size)]
        for batch in train_loader:
            y_batch_LS = batch["target_data"].to(device)  # (B, T, C)
            x_batch_LS = batch["cond_data"].to(device)

            y_batch_LS = y_batch_LS.reshape(y_batch_LS.shape[0], -1)  # (B, T*C)
            x_batch_LS = x_batch_LS.reshape(x_batch_LS.shape[0], -1)  # (B, T*C)
            bs = y_batch_LS.shape[0]

            # for y_batch, x_batch in zip(np.split(y_LS_shuffled, batch_list), np.split(x_LS_shuffled, batch_list)):
            # y_batch_LS = torch.tensor(y_batch).to(device).float()
            # x_batch_LS = torch.tensor(x_batch).to(device).float()
            # bs = x_batch_LS.shape[0]

            # 1. Train the Discriminator
            # Critic wants to maximize : E(C(x)) - E(C(G(z)))
            #             <~> maximize : mean(C(x)) - mean(C(G(z)))
            #             <-> minimize : -{mean(C(x)) - mean(C(G(z)))}

            # Generated samples
            G_LS_samples = gen(
                noise=torch.randn(bs, gen.latent_s).to(device), context=x_batch_LS
            )
            # Compute Discriminator's loss
            loss_D = dis.loss(
                generated_samples=G_LS_samples,
                true_samples=y_batch_LS,
                context=x_batch_LS,
            )
            loss_D_batch += loss_D.detach()
            # Update critic's weight
            opt_dis.zero_grad()
            loss_D.backward()
            opt_dis.step()

            # N_CRITIC update for discriminator while one for generator
            # 2. Train the Generator
            if ((batch_dis_idx + 1) % n_discriminator) == 0:
                # Train Generator
                # Generator has the opposed objective that of critic :
                #      wants to minimize : E(C(x)) - E(C(G(z)))
                #           <-> minimize : - E(C(G(z)))
                #           <-> minimize : -(mean(C(G(z)))
                # Generated samples
                G_LS_samples = gen(
                    noise=torch.randn(bs, gen.latent_s).to(device), context=x_batch_LS
                )
                D_LS = dis(input=G_LS_samples, context=x_batch_LS)
                # Compute generator's loss
                lossG = -torch.mean(D_LS)
                loss_G_batch += lossG.detach()
                # Update generator's weight
                opt_gen.zero_grad()
                lossG.backward()
                opt_gen.step()
                batch_gen_idx += 1

            batch_dis_idx += 1

        # LS loss is the average over all the batch
        loss_D_LS = loss_D_batch / batch_dis_idx
        loss_G_LS = loss_G_batch / batch_gen_idx

        # VS loss
        # D

        G_VS_samples = gen(
            noise=torch.randn(real_dataset.shape[0], gen.latent_s).to(device),
            context=torch.from_numpy(real_cond).to(device),
        )
        loss_D_VS = dis.loss(
            generated_samples=G_VS_samples,
            true_samples=torch.from_numpy(real_dataset).to(device),
            context=torch.from_numpy(real_cond).to(device),
        ).detach()
        # G
        D_VS = dis(input=G_VS_samples, context=torch.from_numpy(real_cond).to(device))
        loss_G_VS = -torch.mean(D_VS).detach()

        # Save NF model when the VS loss is minimal
        loss_list.append([loss_D_LS, loss_G_LS, loss_D_VS, loss_G_VS])

        end = timer()
        time_tot += end - start

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:.0f} Approximate time left : {time_tot / (epoch + 1) * (nb_epoch - (epoch + 1)) / 60:2f} min - D LS loss: {loss_D_LS:4f} G LS loss: {loss_G_LS:4f} D VS loss: {loss_D_VS:4f} G VS loss: {loss_G_VS:4f}",
                end="\r",
                flush=True,
            )
    print("Fitting time_tot %.0f min" % (time_tot / 60))

    return np.asarray(torch.tensor(loss_list, device="cpu")), gen, dis


def build_gan_scenarios(
    val_loader: DataLoader,
    gen,
    max: int = 1,
    gpu: bool = True,
    tag: str = "pv",
    non_null_indexes: list = [],
):
    """
    Build scenarios for a VAE multi-output (VS or TEST sets).
    Scenarios are generated into an array (n_periods, n_s) where n_periods = 24 * n_days
    :return: scenarios (n_periods, n_s)
    """

    # to assign the data to GPU with .to(device) on the data
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    gen.to(device)
    time_tot = 0.0
    scenarios = []

    with torch.no_grad():
        for batch in val_loader:
            start = timer()
            # sample nb_scenarios per day of the VS or TEST sets
            conditions = batch["cond_data"]

            x = conditions.reshape(conditions.shape[0], -1)
            print(x.shape)
            predictions = gen.sample(n_s=conditions.shape[0], x_cond=x[0, :])

            scenarios_tmp = predictions
            scenarios.append(
                scenarios_tmp.transpose()
            )  # list of arrays of shape (24, n_s)
            end = timer()
            time_tot += end - start
        print("Scenario generation time_tot %.1f min" % (time_tot / 60))
        return np.concatenate(scenarios, axis=0)  # shape = (24*n_days, n_s)


def plot_GAN_loss(loss: np.array, ylim: list, dir_path: str, name: str):
    """
    Plot the loss vs epoch.
    """
    FONTSIZE = 10
    nb_epoch = loss.shape[0]

    plt.figure()
    plt.plot(loss[:, 0], label="D LS")
    plt.plot(loss[:, 1], label="G LS")
    plt.plot(loss[:, 2], label="D VS")
    plt.plot(loss[:, 3], label="G VS")
    plt.hlines(y=0, xmin=0, xmax=nb_epoch)
    plt.xlabel("epoch", fontsize=FONTSIZE)
    plt.ylabel("ll loss", fontsize=FONTSIZE)
    plt.tick_params(axis="both", labelsize=FONTSIZE)
    plt.xlim(0, nb_epoch)
    plt.ylim(ylim[0], ylim[1])
    plt.title(name + "#LS ")
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dir_path + name + ".pdf")
    plt.show()


# ------------------------------------------------------------------------------------------------------------------
# GEFcom IJF_paper case study
# Solar track: 3 zones
# Wind track: 10 zones
# Load track: 1 zones
# 50 days picked randomly per zone for the VS and TEST sets
#
# A multi-output wasserstein GAN with gradient penalty:
# Generator = a linear generator
# Discriminator = a wasserstein discriminator
# ------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    config = Config()
    tag = "load"  # pv, wind, load
    gpu = True  # put False to use CPU
    print(f"Using gpu: {torch.cuda.is_available()} ")
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir_path = "train/WGANGP/"

    if not os.path.isdir(dir_path):  # test if directory exist
        os.makedirs(dir_path)

    # ------------------------------------------------------------------------------------------------------------------
    # Built the LS, VS, and TEST sets
    # ------------------------------------------------------------------------------------------------------------------
    dataset = "elektro"  # or LCL
    base_directory = os.getcwd()
    file_path_train = base_directory + f"/data/train_{dataset}.csv"
    file_path_val = base_directory + f"/data/val_{dataset}.csv"

    mode = input("Enter mode (train/eval): ").strip().lower()

    if mode == "train":
        print("Starting Training...")
        # Train just diffusion
        config.train["batch_size"] = 64
        config.train["num_epochs"] = 500  # 50000
        nb_epoch = config.train["num_epochs"]
        train_loader = build_diffusion_loaders(
            file_path_train, config, normalisation="minmax"
        )  # splits xtrain y trrain into ls and vs?
        # when processed with batch 64 it's going to crash due to organise customer function
        val_loader = build_diffusion_loaders(
            file_path_val, config, split="validation", train_data=train_loader.dataset
        )  # splits xtrain y trrain into ls and vs?

        ylim_loss = [-40, 10]
        ymax_plf = 2
        ylim_crps = [0, 5]
        nb_zones = 1
        indices = []

        # ------------------------------------------------------------------------------------------------------------------
        # Scale the LS, VS, and TEST sets
        # ------------------------------------------------------------------------------------------------------------------

        # WARNING: use the scaler fitted on the TRAIN LS SET !!!!
        # x_LS_scaled, y_LS_scaled, x_VS_scaled, y_VS_scaled, y_LS_scaler = scale_data_multi(x_LS=df_x_LS.values, y_LS=df_y_LS.values, x_VS=df_x_VS.values, y_VS=df_y_VS.values, x_TEST=df_x_TEST.values, y_TEST=df_y_TEST.values)

        non_null_indexes = list(np.delete(np.asarray([i for i in range(24)]), indices))

        # ------------------------------------------------------------------------------------------------------------------
        # Set the model
        # ------------------------------------------------------------------------------------------------------------------

        n_s = 100
        N_q = 99

        # Set the torch seed for result reproducibility
        torch_seed = 0
        torch.manual_seed(torch_seed)

        # Set the VAE configuration
        config_GAN = config.GAN_wasserstein

        # --------------------------------------------------------------------------------------------------------------
        # Build the VAE
        # --------------------------------------------------------------------------------------------------------------
        name = tag + "_" + config_GAN["name"] + "_" + str(torch_seed)
        print(name)

        config_GAN["Adam_args"] = {
            "betas": config_GAN["betas"],
            "lr": config_GAN["learning_rate"],
            "weight_decay": config_GAN["weight_decay"],
        }

        # Instance critic neural network: discriminator using Wasserstein distance estimate
        dis = Discriminator_wassertein(
            latent_s=config_GAN["latent_s"],
            cond_in=config_GAN["cond_in"],
            in_size=config_GAN["in_size"],
            gen_w=config_GAN["gen_w"],
            gen_l=config_GAN["gen_l"],
            lambda_gp=config_GAN["lambda_gp"],
            gpu=gpu,
        )
        dis.weights_initialize(mean=0.0, std=0.02)
        dis.train()
        # Instance generator neural network
        gen = Generator_linear(
            latent_s=config_GAN["latent_s"],
            cond_in=config_GAN["cond_in"],
            in_size=config_GAN["in_size"],
            gen_w=config_GAN["gen_w"],
            gen_l=config_GAN["gen_l"],
            gpu=gpu,
        )
        gen.weights_initialize(mean=0.0, std=0.02)
        gen.train()

        # Instance optimizers
        opt_D = torch.optim.Adam(
            dis.parameters(),
            lr=config_GAN["learning_rate"],
            betas=config_GAN["betas"],
            weight_decay=config_GAN["weight_decay"],
        )
        opt_G = torch.optim.Adam(
            gen.parameters(),
            lr=config_GAN["learning_rate"],
            betas=config_GAN["betas"],
            weight_decay=config_GAN["weight_decay"],
        )

        # --------------------------------------------------------------------------------------------------------------
        # Fit the GAN
        # --------------------------------------------------------------------------------------------------------------
        print(f"Fit GAN with {nb_epoch} epochs")
        training_time = 0.0
        start = timer()
        loss, gen, dis = fit_gan_wasserstein(
            nb_epoch=nb_epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            gen=gen,
            dis=dis,
            opt_gen=opt_G,
            opt_dis=opt_D,
            n_discriminator=config_GAN["n_discriminator"],
            gpu=gpu,
        )

        end = timer()
        training_time += end - start
        print(f"Training time {training_time:.2f} s")
        epoch_min_D = np.nanargmin(loss[:, 2])
        epoch_min_G = np.nanargmin(loss[:, 3])
        print(
            f"epoch {epoch_min_D} loss D VS is min = {loss[epoch_min_D, 2]:.2f} epoch {epoch_min_G} loss G VS is min = {loss[epoch_min_G, 3]:.2f}"
        )

        torch.save(loss, dir_path + "loss_" + name + "_" + dataset)
        torch.save(gen, dir_path + name + "_" + dataset)
        print(f"Checkpoint saved to {dir_path + 'loss_' + name + '_' + dataset}")
        # --------------------------------------------------------------------------------------------------------------
        # Plot loss function
        # --------------------------------------------------------------------------------------------------------------
        plot_GAN_loss(loss=loss, ylim=ylim_loss, dir_path=dir_path, name="ll_" + name)
    if mode == "eval":
        dir_path = "train/WGANGP/load_WGANGP_1_0"
        config.train["batch_size"] = 158
        max_cap = 1
        # Scenarios are generated into a dict of length nb days (#VS or # TEST sizes)
        # Each day of the dict is an array of shape (n_scenarios, 24)
        generation_time = 0.0
        start = timer()
        train_dataset_diff = build_diffusion_loaders(
            file_path_train, config, normalisation="minmax"
        )
        val_dataset_diff = build_diffusion_loaders(
            file_path_val,
            config,
            split="validation",
            train_data=train_dataset_diff.dataset,
        )

        real_data_list = []
        for batch in val_dataset_diff:
            x0 = batch["target_data"].to(device)  # Individaul
            real_data_list.append(x0.detach().cpu().numpy())

        real_dataset = np.concatenate(real_data_list, axis=0)

        B, C, T = real_dataset.shape
        num_days = B // 158

        real_dataset = real_dataset.reshape(num_days, 158, T, 1)
        real_dataset = real_dataset.transpose(0, 2, 1, 3).squeeze(-1)  # B, T, C

        indices = []
        non_null_indexes = list(np.delete(np.asarray([i for i in range(24)]), indices))
        config_GAN = config.GAN_wasserstein
        # Instance generator neural network
        gen = Generator_linear(
            latent_s=config_GAN["latent_s"],
            cond_in=config_GAN["cond_in"],
            in_size=config_GAN["in_size"],
            gen_w=config_GAN["gen_w"],
            gen_l=config_GAN["gen_l"],
            gpu=gpu,
        )
        gen.weights_initialize(mean=0.0, std=0.02)
        gen.eval()
        gen = torch.load(dir_path, weights_only=False)

        # Sample MODEL
        generated_data = build_gan_scenarios(
            val_loader=val_dataset_diff,
            gen=gen,
            max=max_cap,
            gpu=gpu,
            tag=tag,
            non_null_indexes=non_null_indexes,
        )

        print(generated_data.shape)

        generated_dataset = reverse_load(
            generated_data.reshape(num_days, T, 158).transpose(0, 2, 1),
            train_dataset_diff,
        )  # C*T, B
        print(generated_dataset.shape)
        generated_dataset = generated_dataset.transpose(0, 2, 1)  # B, T, C
        print(generated_dataset.shape)

        print(real_dataset.shape)
        end = timer()
        generation_time += end - start
        print(f"Generation time (LS, VS, TEST) {generation_time:.2f} s")
        power_values = {}

        power_values["wgan"] = {"real": real_dataset, "synth": generated_dataset}

        plot_mean_data(power_values, label=1)
