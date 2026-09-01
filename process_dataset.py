import numpy as np
import pandas as pd
import torch
import ast
from torch.utils.data import Dataset, DataLoader
from visualisations import*
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime


def extract_calendar_features(data, schema):
    # Ensure Date and Time columns are treated as strings before combining
    #breakpoint()
    try:
        bool(datetime.strptime(data[schema.date][0], "%Y-%m-%d"))
        data['datetime'] = pd.to_datetime(data[schema.date].astype(str) + ' ' + data[schema.time].astype(str), format='%Y-%m-%d %H:%M:%S') #%Y-%m-%d or 
    except:
        bool(datetime.strptime(data[schema.date][0], "%d/%m/%Y"))
        data['datetime'] = pd.to_datetime(data[schema.date].astype(str) + ' ' + data[schema.time].astype(str), format='%d/%m/%Y %H:%M:%S')

    dow = data['datetime'].dt.dayofweek.astype(int).values.reshape(-1, 1)  # 0=Monday, 6=Sunday

    encoder = OneHotEncoder(categories=[range(7)], sparse_output=False, dtype=np.float32)
    day_one_hot = encoder.fit_transform(dow)
    
    # Create names
    dow_col_names = [f"dow_{i}" for i in range(7)]
    
    # Assign back to original Dataframe
    data[dow_col_names] = day_one_hot


    month = data['datetime'].dt.month.astype(int).values.reshape(-1, 1)   

    encoder = OneHotEncoder(categories=[range(1, 13)], sparse_output=False, dtype=np.float32)
    month_one_hot = encoder.fit_transform(month)
    
    # Create 12 names
    month_col_names = [f"month_{i}" for i in range(1, 13)]
    data[month_col_names] = month_one_hot


    data = data.drop(columns=['datetime'])
    
    return data, dow_col_names, month_col_names

class Schema:
    # date/time
    date = "Date"
    time = "Time"
    # weather
    temp = "temperature"
    ghi = "clear_sky_ghi"
    rain = "rainfall"
    wind = "wind_speed"
    humidity = "humidity"
    # calendar
    dow = "Day_of_Week"       
    month = "Month"   
    holiday = "Holiday"
    # targets
    #probabilities = ["Power(kW)_calendar", "Power(kW)_weather"]
    
    # appliance 
    loads = "Power(kW)" #[f"{i}" for i in range(1, 5) if i not in {14, 157, 159}]
    aggregate = ["Aggregate", "target_load"]
    aggregate_inf = ["target_load"]

class DiffusionDataset(Dataset):
    def __init__(self, file_path, schema, batch_size, seq_len=48, stride=48, split="train", normalisation="lognorm", train_data=None, selected_date = None):
        data = pd.read_csv(file_path)
        print(data)
        self.seq_len = seq_len
        self.stride = stride
        self.split = split
        self.sch = schema
        self.normalisation = normalisation
        target_cols = self.sch.loads
        self.date = self.sch.date

        if split in ('validation', 'inference'):

            if selected_date is not None:
                print(data[self.sch.date])
                print(selected_date)
                data = data[((pd.to_datetime(data[self.sch.date], dayfirst=True) >= str(selected_date[0])) & \
                    (pd.to_datetime(data[self.sch.date], dayfirst=True) < str(selected_date[-1])))].reset_index(drop=True)
                print(data)
            
            #remainder = 7 * seq_len

            #data = data.iloc[remainder:].reset_index(drop=True)
            if split == 'validation':
                customer_data = data[target_cols]


            #repeat each day 158 times
            batches = [data.iloc[i:i+seq_len] for i in range(0, len(data), seq_len) for _ in range(batch_size)]
            data = pd.concat(batches, ignore_index=True)
        

        #Find TE features
        data, dow_col, month_col = extract_calendar_features(data, self.sch)

        #Conditions

        exog_col_w = [
            self.sch.temp, self.sch.ghi, self.sch.wind, self.sch.rain, self.sch.humidity  # Continuous (Weather)
        ]
        exog_col_cal = dow_col + month_col + [self.sch.holiday]

        exog_weather = data[exog_col_w].values.astype(np.float32)
        exog_calendar  = data[exog_col_cal].values.astype(np.float32)
        #exog_stats = data[self.sch.probabilities].values.astype(np.float32)

        ####

        if split == 'train':
            #Target loads
            target_cols = self.sch.loads
            self.num_targets = len(target_cols)
            target_data = data[target_cols].values.astype(np.float32)
            print(f"Target data shape: {target_data.shape}")
            #Normlaising parameters
            self.exog_max = np.max(exog_weather, axis=0, keepdims=True)
            if normalisation == "lognorm":
                print(normalisation)
                self.mean = np.mean(target_data, axis=0, keepdims=True)
                self.std = np.std(target_data, axis=0, keepdims=True)
                
                #self.mean_load_stats = np.mean(exog_stats, axis=0, keepdims=True)
                #self.std_load_stats = np.std(exog_stats, axis=0, keepdims=True)

                target_data = self.log_norm_train(target_data, self.mean, self.std)
                #exog_stats = self.log_norm_train(exog_stats,  self.mean_load_stats, self.std_load_stats)

            elif normalisation == "minmax":
                print(normalisation)
                self.max_load = np.max(target_data, axis=0, keepdims=True)
                self.min_load = np.min(target_data, axis=0, keepdims=True)

                #self.max_load_stats = np.max(exog_stats, axis=0, keepdims=True)
                #self.min_load_stats = np.min(exog_stats, axis=0, keepdims=True)

                target_data = self.minmax_norm_train(target_data, self.max_load, self.min_load)
                #exog_stats = self.minmax_norm_train(exog_stats,  self.max_load_stats, self.min_load_stats)

                self.mean = np.mean(target_data, axis=0, keepdims=True)
                self.std = np.std(target_data, axis=0, keepdims=True)
                
                #self.mean_load_stats = np.mean(exog_stats, axis=0, keepdims=True)
                #self.std_load_stats = np.std(exog_stats, axis=0, keepdims=True)

                
                target_data = (target_data - self.mean) / (self.std + 1e-8)
                #exog_stats = (exog_stats - self.mean_load_stats) / (self.std_load_stats + 1e-8)

            
            # Apply Norm
            exog_weather = self.max_norm(exog_weather, self.exog_max)

            self.windows_target = make_windows(target_data[:, np.newaxis] , seq_len, stride)

        elif split == 'validation':
            #Target loads
            #breakpoint()
            customer_data = customer_data.apply(ast.literal_eval)
            target_data =np.concatenate(customer_data).astype(np.float32) #need a more permanent fix
            target_data = organise_customers(customer_data, seq_len, batch_size)

            
            #target_data = data[target_cols].values.astype(np.float32)
            #Normlaising parameters
            self.exog_max = train_data.exog_max

            """
            if normalisation == "lognorm":

                self.mean_load_stats = train_data.mean_load_stats
                self.std_load_stats = train_data.std_load_stats

                exog_stats = self.log_norm_train(exog_stats,  self.mean_load_stats, self.std_load_stats)

            elif normalisation == "minmax":

                self.max_load_stats = train_data.max_load_stats
                self.min_load_stats = train_data.min_load_stats
                

                exog_stats = self.minmax_norm_train(exog_stats, self.max_load_stats, self.min_load_stats)
                self.mean_load_stats = np.mean(exog_stats, axis=0, keepdims=True)
                self.std_load_stats = np.std(exog_stats, axis=0, keepdims=True)
           
                exog_stats = (exog_stats - self.mean_load_stats) / (self.std_load_stats + 1e-8)
            """

            # Apply Norm
            exog_weather = self.max_norm(exog_weather, self.exog_max)
            
            self.windows_agg   = make_windows(data[self.sch.aggregate].values.astype(np.float32), seq_len, stride)
            self.windows_target = make_windows(target_data[:, np.newaxis] , seq_len, stride)

        elif split == 'inference':
            #Normlaising parameters
            self.exog_max = train_data.exog_max

            """
            if normalisation == "lognorm":

                self.mean_load_stats = train_data.mean_load_stats
                self.std_load_stats = train_data.std_load_stats

                #exog_stats = self.log_norm_train(exog_stats,  self.mean_load_stats, self.std_load_stats)

            elif normalisation == "minmax":

                self.max_load_stats = train_data.max_load_stats
                self.min_load_stats = train_data.min_load_stats

                #exog_stats = self.minmax_norm_train(exog_stats, self.max_load_stats, self.min_load_stats)
            """
            # Apply Norm
            exog_weather = self.max_norm(exog_weather, self.exog_max)
            self.windows_agg   = make_windows(data[self.sch.aggregate_inf].values.astype(np.float32), seq_len, stride)

        exog_data = np.concatenate([exog_weather, exog_calendar], axis=1)

        self.windows_cond   = make_windows(exog_data, seq_len, stride)
        
    def log_norm_train(self, data, mean, std):
       
        data = np.log1p(data)  # log(1 + x) to avoid log(0) errors
        data = (data - mean) / (std + 1e-8)

        return data

    def minmax_norm_train(self, data, max_load, min_load):

        print(f"Max load: {max_load}, Min load: {min_load}")
        data = (data - min_load) / (max_load - min_load + 1e-8)
        
        return data

    def max_norm(self, data, max_val):
        return data / (max_val + 1e-8)

    def __len__(self):
        return len(self.windows_cond)

    def __getitem__(self, idx):
        if self.split == 'train':
            target_seq = torch.tensor(self.windows_target[idx]).float().transpose(0, 1)
            cond_seq   = torch.tensor(self.windows_cond[idx]).float().transpose(0, 1)  
          
            return {
                "target_data": target_seq,  
                "cond_data": cond_seq,       
            }
        elif self.split == 'validation':
            target_seq = torch.tensor(self.windows_target[idx]).float().transpose(0, 1)
            cond_seq   = torch.tensor(self.windows_cond[idx]).float().transpose(0, 1)  
            agg_seq = torch.tensor(self.windows_agg[idx]).float().transpose(0, 1)  

            return {
                "target_data": target_seq,  
                "cond_data": cond_seq,       
                "agg_data": agg_seq,      
            }
        elif self.split == 'inference':
            cond_seq   = torch.tensor(self.windows_cond[idx]).float().transpose(0, 1)  
            agg_seq = torch.tensor(self.windows_agg[idx]).float().transpose(0, 1)  

            return {
                "cond_data": cond_seq,   
                "agg_data": agg_seq,    
            }


def build_diffusion_loaders(file_path, config, split="train", train_data=None, selected_date=None, normalisation = "lognorm"):
    schema = Schema()
    dataset_diff = DiffusionDataset(file_path, schema, config.train["batch_size"], config.seq_len, config.stride, split=split, train_data=train_data, selected_date=selected_date, normalisation=normalisation) #change split for val
    dataloader_diff =   DataLoader(dataset_diff, config.train["batch_size"], shuffle=(split == "train")) #change shuffle and drop_lat

    return dataloader_diff

def make_windows(data, seq_len, stride):
    print(data.shape)
    num_samples, num_features = data.shape
    num_windows = (num_samples - seq_len) // stride + 1
    
    windows = []
    for i in range(num_windows):
        start = i * stride
        end = start + seq_len
        windows.append(data[start:end, :])
        
    return np.stack(windows, axis=0)

def reverse_load(data, data_real):

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy() 

    data_real = data_real.dataset if isinstance(data_real, DataLoader) else data_real

    if data_real.normalisation == "lognorm":

        mean = data_real.mean 
        std = data_real.std
        mean = mean[:, np.newaxis]  # Becomes (1, 158, 1)
        std  = std[:, np.newaxis]   # Becomes (1, 158, 1)
        print(mean)
        data = data * std + mean  # Reverse standardization
        data = np.clip(data, 0, None)
        data = np.expm1(data)  # Reverse log transformation

    elif data_real.normalisation == "minmax":
        max_load = data_real.max_load
        min_load = data_real.min_load

        mean = data_real.mean 
        std = data_real.std
        mean = mean[:, np.newaxis]  # Becomes (1, 158, 1)
        std  = std[:, np.newaxis]   # Becomes (1, 158, 1)
        data = data * std + mean
        data =  np.clip(data, 0, 1)
        data = data * (max_load - min_load + 1e-8) + min_load  # Reverse min-max normalization
        
   

    print(f"Min: {data.min():.4f}, Max: {data.max():.4f}, Mean: {data.mean():.4f}")
    return data

def reverse_load_train(data, data_real, clamp_min = -9.0, clamp_max=9.0):

    # Support both DataLoader and Dataset
    dataset = data_real.dataset if isinstance(data_real, DataLoader) else data_real
    mean = torch.tensor(dataset.mean, dtype=torch.float32, device=data.device)
    std = torch.tensor(dataset.std, dtype=torch.float32, device=data.device)


    data = data * std + mean   

    return torch.expm1(data)


def organise_customers(customer_data, seq_len, batch_size):
    
    days = int(len(customer_data)/seq_len)
    num_targets = [len(customer_data[i*seq_len]) for i in range(days)]
    print(num_targets)
    #Create matrix for n*158, fill missing with 0
    target_data = np.zeros(len(customer_data)*batch_size)
    print(f"This is target data len {target_data.shape}")
    day=0

    for c in range(len(num_targets)): #days
        for j in range(num_targets[c]): #[152, 151..158]
            for i in range(seq_len): #timestep
                target_data[i+day] = customer_data[i+(c*seq_len)][j] # for every timestep save every customer column 
            day+=seq_len
        if (num_targets[c] < batch_size): #when missing fill with 0
            for k in range((batch_size - num_targets[c])*seq_len):
                target_data[day + k ] = 0.0
            day += (batch_size - num_targets[c])*seq_len #set the index right

    return target_data
