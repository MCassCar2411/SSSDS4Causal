from pathlib import Path

import pandas as pd
from config.cfg import Config
from DML.causality import (
    build_diffusion_loaders,
    calculate_sub_impact,
    causality_analysis,
    model_robustness,
    refute_random_common_cause,
)
from training import *

BASE_DIRECTORY: Path = Path.cwd()

# or LCL
DATASET: str = "elektro"

CAUSAL_PATH: Path = BASE_DIRECTORY / f"data/causal_train_{DATASET}.csv"
CAUSAL_PATH_VAL: Path = BASE_DIRECTORY / f"data/causal_val_{DATASET}.csv"

# train 2 year individual ##Update to LCL train_LCL_1
FILE_PATH_TRAIN: Path = BASE_DIRECTORY / f"data/train_{DATASET}.csv"
# test validation 6 months ##Update to LCL if eval
FILE_PATH_VAL: Path = BASE_DIRECTORY / f"data/val_{DATASET}.csv"
# test extremes
FILE_PATH_EXTREMES: Path = BASE_DIRECTORY / f"data/test_{DATASET}_extremes.csv"
FILE_PATH_INFER: Path = BASE_DIRECTORY / "data/conditions_langside.csv"

N_SAMPLES: int = 100
RANDOM_SEED: int = 42


def main():
    set_seed(RANDOM_SEED)
    config: Config = Config()

    # Load training dataset
    inference_data_path: Path = Path(
        "data/inference/gen_data"
    )  # to store synthetic data

    mode: str = (
        input("Enter mode (causality/train/evaluate/generate): ").strip().lower()
    )

    match mode:
        case "causality":
            print("Starting Causal Analysis with DML...")
            size: str = "hour"
            # Load data
            df: pd.DataFrame = pd.read_csv(CAUSAL_PATH)
            treatment: list[str] = ["temperature"]
            controls: list[str] = [
                "temperature",
                "wind_speed",
                "humidity",
                "radiation",
                "clear_sky_ghi",
                "rainfall",
                "month",
                "Holiday",
                "hour",
            ]
            # Eval model
            model_robustness(CAUSAL_PATH, treatment, controls, DATASET)

            # Train
            causality_analysis(df, treatment, controls, size, DATASET, n_iterations=1)

            # Evaluate causal estimates
            refute_random_common_cause(
                DATASET, num_simulations=100, random_state=RANDOM_SEED, df=CAUSAL_PATH
            )

        case "train":
            print("Starting Training...")

            # Train just diffusion
            config.train["batch_size"] = 64
            train_dataset_diff = build_diffusion_loaders(FILE_PATH_TRAIN, config)
            train_diffusion(config, train_dataset_diff, DATASET)

        case "evaluate":
            print("Evaluating Model...")

            # Estimate Causal Impact at substation level
            config.train["batch_size"] = 158
            calculate_sub_impact(
                FILE_PATH_VAL, dataset=DATASET, cal_path=CAUSAL_PATH_VAL, task="val"
            )

            # Evaluate Diffusion model
            evaluate(config, FILE_PATH_TRAIN, FILE_PATH_VAL, DATASET, test_pvalue=False)

        case "generate":
            print("Generating Synthetic Data...")

            # Inference path
            inference_data_path = BASE_DIRECTORY / "data/inference/gen_data/test"

            config.train["batch_size"] = 158  # or n_customers
            generate_samples(
                FILE_PATH_TRAIN,
                config,
                FILE_PATH_EXTREMES,
                inference_data_path=inference_data_path,
                n_samples=N_SAMPLES,
                dataset=DATASET,
            )
        case _:
            print("Please enter 'train', 'evaluate', or 'generate'.")


if __name__ == "__main__":
    main()
