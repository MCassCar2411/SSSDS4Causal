import torch
import numpy as np
import torch.nn as nn
from metrics import*
from network import*


class Diffusion:

    def __init__(self, config):
        self.device = config.train["device"]
        self.seq_length= config.seq_len
        self.batch_size = config.train["batch_size"]
        self.input_dim = config.input_dim #features
        self.t_emb = config.model["t_emb"] #Time embedding

        config_diff = config.diffusion

        self.diffusion_timesteps = config_diff["T"] 

        if config_diff["schedule"] == "quad":
            self.beta = torch.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.diffusion_timesteps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = torch.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.diffusion_timesteps
            )

        self.alpha = 1 - self.beta
        self.alphas_cumprod = torch.cumprod(self.alpha, dim=0)

        # Forward Diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Predict Start from Noise
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

        # Backward Diffusion coefficients
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0) # alpha for previous time step
        self.posterior_mean_coef1 = self.beta * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) # first part of equation with x0
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alpha) / (1.0 - self.alphas_cumprod) # second part with xt
        self.posterior_variance = self.beta * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)  ##third part with error term
        self.model_log_variance = torch.log(torch.clamp(self.posterior_variance, min=1e-20))  # Avoid log(0) log variance for stability, exponentiated during sampling

        # Move all tensors to device
        self.beta = self.beta.to(self.device)
        self.alpha = self.alpha.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)

        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(self.device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(self.device)

        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(self.device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(self.device)
        self.posterior_variance = self.posterior_variance.to(self.device)
        self.model_log_variance = self.model_log_variance.to(self.device)


        self.denoiser = SSSDS4(config) 

    def forward_diffusion(self, x_0, t):

        noise = torch.randn_like(x_0)   # Generate Gaussian noise
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1) ##(batch, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1) ##

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    #Single step reconstruction
    def predict_start_from_noise(self, x_t, noise_pred, t):

        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].reshape(-1, 1, 1)##
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].reshape(-1, 1, 1)##

        x_0_reconstructed = sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise_pred

        return x_0_reconstructed


    # One step reconstruction 
    def reverse_diffusion(self, x_t, pred, t, timestep):
           
        #This is x0-reparametrisation approach
        x_0_pred = self.predict_start_from_noise(x_t, pred, t) #Estimate clean x0 from noisy xt and predicted noise

        #Compute posterior mean combining x0_pred and xt
        posterior_mean_coef1_t = self.posterior_mean_coef1[t].reshape(-1, 1, 1)
        posterior_mean_coef2_t = self.posterior_mean_coef2[t].reshape(-1, 1, 1)

        model_mean = posterior_mean_coef1_t * x_0_pred + posterior_mean_coef2_t * x_t

        #Sample from posterior, add noise except at t=0 to simulate process
        if timestep == 0:
            x_tm1 = model_mean
        else:
            model_log_variance_t = self.model_log_variance[t].reshape(-1, 1, 1)
            noise = torch.randn_like(x_0_pred)  # Sample noise1sps
            x_tm1 = model_mean + torch.exp(0.5 * model_log_variance_t) * noise 

        return x_tm1

        """
        alpha_t = self.alpha[t].reshape(-1, 1, 1)
        posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1)
        sqrt_recip_alpha_t = torch.sqrt(1 / alpha_t)
        sqrt_one_minus_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)

        x_t_scaled = x_t * sqrt_recip_alpha_t
        eps_coef = (1 - alpha_t) / sqrt_one_minus_cumprod_t * sqrt_recip_alpha_t
        model_mean = x_t_scaled - eps_coef * pred

        if timestep == 0:
            x_tm1 = model_mean
        else:
            noise = torch.randn_like(x_t)
            x_tm1 = model_mean + torch.sqrt(posterior_variance_t) * noise

        return x_tm1
        """

        
    def infer(self, denoiser, dataset, real_dataset=None, fixed_noise = False, type="DDPM", p=10):

        generated_data = []
        if fixed_noise:
            x_t_fixed = torch.load("train/x_t.pt")
            j=0

        for batch in dataset:

            conditions = batch["cond_data"].to(self.device)  # (B, T, num_exogenous)
            B, _, _ = conditions.shape
           
            #Fixed noise for evaluation
            if fixed_noise:
                x_t = x_t_fixed[j:j+B]
                j += B

            else:
                
                x_t = torch.randn((B, self.input_dim, self.seq_length), device=self.device)   # Start from pure noise
            
            if type == "DDPM":

                for i in reversed(range(self.diffusion_timesteps)):  
                    #x_t = x_t.detach()
                    t_tensor = torch.full((B,), i, device=self.device, dtype=torch.long)
                   
                    noise_pred = denoiser(x_t, conditions, t_tensor).to(self.device)
                    x_t = self.reverse_diffusion(x_t, noise_pred, t_tensor, i) #one step and stochastis and non-differentiable 
                    x_t = x_t.detach().cuda()
        
            elif type == "Guide":
                #Each batch should be a different customer for the same day
                if batch["agg_data"].ndim == 1:
                    agg_real =  batch["agg_data"][1].to(self.device)
                    
                else:
                    agg_real =  batch["agg_data"][0][0].to(self.device)

                print("This is the real aggregate for this batch: ", agg_real)
                #Sharpness
                eve_ramp_r = torch.diff(agg_real)

                max_limit = None
                num_samples = x_t.shape[0]
                T = x_t.shape[2]
                #agg_generated = torch.zeros((1,T), device=self.device)
                #real_data=train_dataset[ind:ind+batch_size]
                for i in reversed(range(self.diffusion_timesteps)): 
                    #x_t = x_t.detach()
                    x_t.requires_grad_(True) 
                    """
                    for j in range(0, num_samples, 158):
                        x_chunk = x_t[j:j+158]
                        current_batch_size = x_chunk.shape[0]
                        t_tensor = torch.full((current_batch_size,), i, device=self.device, dtype=torch.long)

                        noise_pred = denoiser(x_chunk, conditions[j:j+158], t_tensor).to(self.device)
                        x0_pred = self.predict_start_from_noise(x_chunk, noise_pred, t_tensor)
                        dataset_1 = reverse_load_train(x0_pred, real_dataset)

                        agg_generated += reverse_load_train(x0_pred, real_dataset).sum(dim=0)
                    """
                    t_tensor = torch.full((B,), i, device=self.device, dtype=torch.long)
                    noise_pred = denoiser(x_t, conditions, t_tensor).to(self.device)
                    x0_pred = self.predict_start_from_noise(x_t, noise_pred, t_tensor)
                    dataset_1 = reverse_load_train(x0_pred, real_dataset)
                    agg_generated = dataset_1.sum(dim=0) # (1, T)
                    B, C, T = x0_pred.shape

                    #dataset1 = reverse_load_train(x0_pred, real_dataset)
                    #agg_generated = dataset1.sum(dim=0) # (1, T)

                    #Divide by cardinal points
                    eve_ramp = torch.diff(agg_generated)
                    
                    # Combine the losses 
                    #loss = torch.mean((synth_min - real_min)**2) + torch.mean((synth_max - real_max)**2) + torch.mean((eve_ramp - eve_ramp_r)**2)
                    #loss = torch.mean((agg_generated - agg_real * (158/160))**2) + torch.mean((agg_generated.max() - agg_real.max() * (158/160))**2)
                    #loss = torch.mean((night_through - night_through_r)**2) + torch.mean((day_peak - day_peak_r)**2) + torch.mean((day_through - day_through_r)**2) + torch.mean((eve_peak - eve_peak_r)**2) + torch.mean((eve_ramp - eve_ramp_r)**2) + torch.mean((morning_ramp - morning_ramp_r)**2) + torch.mean((start_synth - start_real)**2)
                    loss = torch.mean((agg_generated - agg_real)**2) + torch.mean((eve_ramp - eve_ramp_r)**2) 
                    #loss =  torch.mean((agg_generated - agg_real)**2) +  torch.mean((agg_generated.max() - agg_real.max())**2) 
                    #breakpoint()
                    #print(loss)
                    #gradient
                    grad = torch.autograd.grad(loss, x_t, retain_graph=True)[0]

                    # Normalize by batch size (158)
                    batch_size = x_t.shape[0]
                    grad = grad / batch_size 

                    scaling_factor = self.sqrt_one_minus_alphas_cumprod[i].view(1, 1, 1)

                    # Calculate the adjustment vector separately for debugging
                    adjustment = p * scaling_factor * grad 
                    noise_guided = noise_pred + adjustment #opposite sign for loss

                    x_t = self.reverse_diffusion(x_t, noise_guided, t_tensor, i) #one step and stochastis and non-differentiable 
                    x_t = x_t.detach().cuda()
                    

            generated_data.append(x_t)

        data = torch.cat(generated_data, dim=0)  

        return data




