import os
from tqdm import tqdm
from process_dataset import*
from training import*
from DML.causality import*
from config.cfg import Config
from multiprocessing import Pool
base_directory = os.getcwd()
print(base_directory)

dataset = "elektro" #or LCL

causal_path = base_directory + f"/data/causal_train_{dataset}.csv"
causal_path_val = base_directory + f"/data/causal_val_{dataset}.csv"

file_path_train = base_directory + f"/data/train_{dataset}.csv" # train 2 year individual ##Update to LCL train_LCL_1
file_path_val = base_directory + f"/data/val_{dataset}.csv" #test validation 6 months ##Update to LCL if eval
file_path_extremes = base_directory + f"/data/test_{dataset}_extremes.csv" #test extremes

file_path_infer = base_directory + f"/data/conditions_langside.csv" #test extremes


if __name__ == "__main__":

     # Initialize
    set_seed(42)
    config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training dataset
    inference_data_path = f"data/inference/gen_data" # to store synthetic data 

    mode = "evaluate"
        
    if mode == "causality":
        print("Starting Causal Analysis with DML...")
        group = False
        if group:
            SPEN_csv = pd.read_csv(base_directory + f"/data/ENSIGN_demo.csv")
            group_sizes = SPEN_csv['Needed'].values
            #Global
            for size in group_sizes:

                causal_path = base_directory + f"/data/{size}_train_ENSIGN.csv"
                estimate_CATE_general(causal_path, size)
            """
            ##Train causal model
            for size in group_sizes:

                causal_path = base_directory + f"/data/{size}_train_ENSIGN.csv"
                causal_path_val = base_directory + f"/data/{size}_test_ENSIGN.csv"
                causality_analysis(causal_path, size, config)
                #Evaluate causal estimand model
                evaluate_causality(causal_path_val, size, config)

            """
        else:
            csv_directory =  base_directory + f"/results/causality"
            """ 
            #To train PEET
            starmap_parameters = assemble_feeder_data(causal_path, csv_directory)

            with Pool(processes=3, maxtasksperchild=2) as p:
                all_results_dfs: list[pd.DataFrame] =   p.starmap(func=estimate_CATE_control, iterable=starmap_parameters)
           
            #To evaluate
            run_cross_feeder(causal_path, csv_directory)
            """
            
            causal_path_val = causal_path_val
            size= "hour"
            # Load data
            df = pd.read_csv(causal_path)
            treatment = ['temperature']
            controls = ['temperature', 'wind_speed', 'humidity', 'radiation', 'clear_sky_ghi', 'rainfall', 'month', 'Holiday', 'hour']
            #Eval model 
            #model_robustness(causal_path, treatment, controls, dataset)
            #Train

            #causality_analysis(df, treatment, controls, size, dataset, n_iterations=1)

            #Evaluate causal estimates
            
            rate_change = refute_random_common_cause(dataset, num_simulations=100, random_state=42, df=causal_path)

            


    elif mode == "train":
        print("Starting Training...")

        #Train just diffusion
        config.train["batch_size"] = 64

        train_dataset_diff = build_diffusion_loaders(file_path_train, config)      

        train_diffusion(config, train_dataset_diff, dataset)

    elif mode == "evaluate":

        print("Evaluating Model...")
        #Estimate Causal Impact at substatio level
        config.train["batch_size"] = 158
        #calculate_sub_impact(causal_path_val, size=config.train["batch_size"], cal_path = causal_path_val, task="val")
        
        calculate_sub_impact(file_path_val, dataset=dataset,  cal_path = causal_path_val, task="val")
        

        #Evaluate Diffusion model          
        evaluate(config, file_path_train, file_path_val, dataset, test_pvalue=False)

    elif mode == "generate":

        print("Generating Synthetic Data...")

        #Inference path
        inference_data_path=base_directory+"/data/inference/gen_data/test"
        csv_path = file_path_extremes

        #If substation
        #csv_path = file_path_extremes
        substation_path=base_directory + f"/data/ENSIGN_demo.csv"

        #if eval
        #csv_path = file_path_val
        #csv_path = base_directory + "/data/157_conditions.csv"
        #substation_path=None


        n_samples = 100
        #n_customers = int(input("Enter number of customer samples to generate: "))
        config.train["batch_size"] = 158 #or n_customers
        #calculate_sub_impact(csv_path, causal_path_val, label="infer")
        #calculate_sub_impact(file_path_extremes, causal_path_val)#

        generated_data = generate_samples(file_path_train, config, csv_path, inference_data_path = inference_data_path, n_samples = n_samples, dataset=dataset, csv_substation=None)


    else:

        print("Please enter 'train', 'evaluate', or 'generate'.")