import csv
import os

import pandas as pd
import yaml
from prettytable import PrettyTable

import wandb


class ResultHandler:
    def __init__(self, result_file, log_wandb=False, keep_existing=False):
        self.result_path = result_file
        self.log_wandb = log_wandb

        # if not keep_existing and os.path.exists(result_path):
        #     os.remove(result_path)

    def log_result(self, results, evaluation_params):
        """
        Log a new result to the CSV file.

        Args:
            results (dict): All results in one dict.
            evaluation_params (EvaluationParams): The parameters as a dataclass instance.
        """

        # create a new dataframe to collect all the results
        out = pd.Series()

        # 1. add the image name and type
        out["image_name"] = results["image_name"]
        out["type"] = evaluation_params.type

        # 2. add the rest of the evaluation parameters
        keys = evaluation_params.__dict__.keys()
        for key in keys:
            # skip the image name and type
            if key == "image_name" or key == "type":
                continue
            out[key] = evaluation_params.__dict__[key]

        # 3. add the results
        out["duration"] = results["duration"]
        out["are"] = results["are"]
        out["precision"] = results["precision"]
        out["recall"] = results["recall"]
        out["f1"] = results["f1"]
        out["jaccard"] = results["jaccard"]
        out["jaccard_cellpose"] = results["jaccard_cellpose"]

        # Log to W&B
        if self.log_wandb:
            wandb.log(out.to_dict())

        # Ensure consistent fieldnames (in the same order as OrderedDict keys)
        fieldnames = list(out.keys())

        # Check if file exists
        file_exists = os.path.exists(self.result_path)

        # Write to CSV file
        with open(self.result_path, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists or os.stat(self.result_path).st_size == 0:
                writer.writeheader()  # Write header only if file is new or empty
            writer.writerow(out.to_dict())

        self.print_results()

    def print_results(self):
        """Print the results as a pretty table."""
        if not os.path.exists(self.result_path):
            print("No results to display.")
            return

        with open(self.result_path, mode="r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            table = PrettyTable()
            table.field_names = reader.fieldnames
            for row in reader:
                formatted_row = [
                    (
                        f"{float(row[field]):.3f}"
                        if field
                        in [
                            "are",
                            "precision",
                            "recall",
                            "f1",
                            "duration",
                            "jaccard",
                            "jaccard_cellpose",
                        ]
                        else row[field]
                    )
                    for field in reader.fieldnames
                ]
                table.add_row(formatted_row)
            print(table)


def print_best_config_per_image(file_path, metric="jaccard"):
    """
    Print the best configuration for each unique image_name and type combination based on the specified metric.
    Save the best configurations to an output JSON file.

    Parameters:
        file_path (str): Path to the CSV file containing experiment results.
        metric (str): The column name of the metric to evaluate (default is 'jaccard').
        output_file (str): Path to save the best configurations (default is 'best_configs.json').
    """
    # Load data

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return

    # Ensure the metric column exists
    if metric not in df.columns:
        print(f"Error: '{metric}' is not a valid column in the file.")
        print("Available columns:", ", ".join(df.columns))
        return

    # Round the metric column to 2 decimal places
    df[metric] = df[metric].round(2)

    # Find the best configuration per image_name and type based on the metric
    best_configs = df.loc[df.groupby(["image_name", "type"])[metric].idxmax()]

    # Sort for better readability
    best_configs = best_configs.sort_values(by=["image_name", "type"])

    # Create output directory in the same folder as the input file
    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save each configuration to a separate YAML file
    excluded_columns = [
        "image_name",
        "type",
        "duration",
        "are",
        "precision",
        "recall",
        "f1",
        "jaccard",
        "jaccard_cellpose",
        "jaccard",
    ]
    for _, row in best_configs.iterrows():
        config = {
            col: row[col] for col in best_configs.columns if col not in excluded_columns
        }
        file_name = f"{row['image_name']}_{row['type']}_config.yaml"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "w") as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)
        print(f"Saved configuration to {file_path}")

    # Print results
    print(f"Best configurations based on '{metric}':\n")
    for _, row in best_configs.iterrows():
        print(f"Image: {row['image_name']}")
        print(f"  Type: {row['type']}")
        print(f"  Best {metric}: {row[metric]}")
        print("  Configuration:")
        for col in df.columns:
            if col not in ["image_name", "type", metric]:
                print(f"    {col}: {row[col]}")
        print("-")
