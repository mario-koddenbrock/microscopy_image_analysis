import csv
import os
from collections import OrderedDict
from dataclasses import asdict

from prettytable import PrettyTable
import pandas as pd
import json


class ResultHandler:
    def __init__(self, result_file, keep_existing=False):
        self.result_path = result_file

        # if not keep_existing and os.path.exists(result_path):
        #     os.remove(result_path)


    def log_result(self, image_name, evaluation_params, duration, are, precision, recall, f1):
        """
        Log a new result to the CSV file.

        Args:
            image_name (str): Name of the image.
            evaluation_params (EvaluationParams): The parameters as a dataclass instance.
            are (float): Average Relative Error.
            precision (float): Precision metric.
            recall (float): Recall metric.
            f1 (float): F1 score.
        """
        # Convert dataclass to dictionary and add additional metrics
        properties = asdict(evaluation_params)
        properties.update({
            "duration": duration,
            "are": are,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        # Ensure image_name is the first column
        ordered_properties = OrderedDict([("image_name", image_name)])
        ordered_properties.update(properties)

        # Ensure consistent fieldnames (in the same order as OrderedDict keys)
        fieldnames = list(ordered_properties.keys())

        # Check if file exists
        file_exists = os.path.exists(self.result_path)

        # Write to CSV file
        with open(self.result_path, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists or os.stat(self.result_path).st_size == 0:
                writer.writeheader()  # Write header only if file is new or empty
            writer.writerow(ordered_properties)

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
                    f"{float(row[field]):.2f}" if field in ['are', 'precision', 'recall', 'f1', 'duration'] else row[field] for
                    field in reader.fieldnames]
                table.add_row(formatted_row)
            print(table)




def print_best_config_per_image(file_path, metric='f1', output_file='best_configs.json'):
    """
    Print the best configuration for each unique image_name and type combination based on the specified metric.
    Save the best configurations to an output JSON file.

    Parameters:
        file_path (str): Path to the CSV file containing experiment results.
        metric (str): The column name of the metric to evaluate (default is 'f1').
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
        print("Available columns:", ', '.join(df.columns))
        return

    # Round the metric column to 2 decimal places
    df[metric] = df[metric].round(2)

    # Find the best configuration per image_name and type based on the metric
    best_configs = df.loc[df.groupby(['image_name', 'type'])[metric].idxmax()]

    # Sort for better readability
    best_configs = best_configs.sort_values(by=['image_name', 'type'])

    # Convert DataFrame to dictionary for saving as JSON
    best_configs_dict = best_configs.to_dict(orient='records')

    # Save the best configurations to a JSON file
    with open(output_file, 'w') as f:
        json.dump(best_configs_dict, f, indent=4)
    print(f"Best configurations have been saved to '{output_file}'.")

    # Print results
    print(f"Best configurations based on '{metric}':\n")
    for _, row in best_configs.iterrows():
        print(f"Image: {row['image_name']}")
        print(f"  Type: {row['type']}")
        print(f"  Best {metric}: {row[metric]}")
        print("  Configuration:")
        for col in df.columns:
            if col not in ['image_name', 'type', metric]:
                print(f"    {col}: {row[col]}")
        print("-")
