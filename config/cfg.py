import torch


num_customers = 1       
num_exogenous = 7 + 1 + 12 + 5 

# The model sees a total width of (Customers + Conditions)
input_dim = num_customers 
seq_length = 48            
#Benchamrk parameters

timegan_config = {
    "hidden_dim": 512,      # unified hidden dimension
    "num_layer": 3,         # layers for GRUs
    "lr": 1e-3,
    "seed": 42 #add to other?

}
#Wgan parameters
GAN_wasserstein = {
    'name': 'WGANGP_1',
    'cond_in':num_exogenous*48,
    'in_size': input_dim*48,
    'latent_s': 256,
    'gen_w': 256 * 4,
    'gen_l': 2,
    'n_discriminator': 5,
    'lambda_gp': 10,
    'betas': (0.0, 0.9),
    'weight_decay': 1 * 10 ** (-4),
    'learning_rate': 1 * 10 ** (-3),
 }

# DIFFUSION PARAMETERS 
diffusion_config = {
    "T": 200,              
    "beta_start": 1e-4,   
    "beta_end": 0.02,       
    "schedule": "linear",    
}


# Denoiser  architecture
model_config = {
    "t_emb": 128,            # Dimension for time-step embeddings

    #SSSDS4
    "in_channels": input_dim,
    "cond_channel": num_exogenous, 
    "res_channels": 128, #back to 256?
    "skip_channels": 128,
    "out_channels": input_dim, # Output shape matches Input
    "num_res_layers": 12, #36
    "diffusion_step_embed_dim_in": 128,
    "diffusion_step_embed_dim_mid": 512,
    "diffusion_step_embed_dim_out": 512,
    "s4_lmax": seq_length,
    "s4_d_state": 64,
    "s4_dropout": 0.0,
    "s4_bidirectional": True, #whether model looks at future
    "s4_layernorm": True
}


# Training parameters
train_config = {
    "pipeline": True,      
    "num_epochs": 200,     
    "batch_size": 64,  #update for training vs eval/infer
    "lr": 2e-4,            # Learning Rate
    "weight_decay": 1e-6,  # Helps prevent overfitting on small datasets
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


# Inference and eval 
eval_config = {
    "epoch_id": 200,       # Load the weights from this epoch
    "n_samples": 50,       # How many distinct "possible days" to generate per input
    "output_dir": "./results/"
}


class Config:
    def __init__(self):
        self.input_dim = input_dim
        self.num_targets = num_customers
        self.num_cond = num_exogenous
        self.seq_len = seq_length
        self.stride = seq_length
        self.diffusion = diffusion_config
        self.model = model_config
        self.train = train_config
        self.eval = eval_config


        self.timegan = timegan_config
        self.GAN_wasserstein = GAN_wasserstein
        


