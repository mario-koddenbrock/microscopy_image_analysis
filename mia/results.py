import csv
import os


class ResultHandler:
    def __init__(self, result_path, fieldnames):
        self.result_path = result_path
        self.fieldnames = fieldnames
        self.existing_results = []


    def _load_existing_results(self):
        """Load existing results from the CSV file into a set."""
        if os.path.exists(self.result_path):
            with open(self.result_path, mode="r", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    self.existing_results.append(self.get_result_key_from_row(row))

    def log_result(self, config, jaccard, f1_score):
        """Log a new result to the CSV file."""
        result_key = self.get_result_key_from_params(config)
        if result_key in self.existing_results:
            return  # Skip duplicate results

        # Build the result dictionary
        result = self.build_result_dict(config, jaccard, f1_score)

        with open(self.result_path, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            if os.stat(self.result_path).st_size == 0:
                writer.writeheader()
            writer.writerow(result)
        self.existing_results.append(result_key)

    def is_result_present(self, config):
        """Check if a result already exists."""
        current_result_key = self.get_result_key_from_params(config)
        # print(current_result_key)
        return current_result_key in self.existing_results

    def build_result_dict(self, params, jaccard, f1_score):
        """Build the result dictionary for logging."""
        result = {
            "model_name": params["model_name"],
            "channel_segment": params["channel_segment"],
            "channel_nuclei": params["channel_nuclei"],
            "channel_axis": params["channel_axis"],
            "invert": params["invert"],
            "normalize": params["normalize"],
            "diameter": params["diameter"],
            "do_3D": params["do_3D"],
            "jaccard": jaccard,
            "f1_score": f1_score,
        }

        return result

    def get_result_key_from_params(self, params):
        """Return a unique identifier for a params dict."""
        return (
            params["model_name"],
            params["channel_segment"],
            params["channel_nuclei"],
            params["channel_axis"],
            params["invert"],
            params["normalize"],
            params["diameter"],
            params["do_3D"],
        )

    def get_result_key_from_row(self, row):
        """Return a unique identifier for a row in the CSV file."""
        return (
            row["model_name"],
            eval(row["channel_segment"]),
            eval(row["channel_nuclei"]),
            eval(row["channel_axis"]),
            eval(row["invert"]),
            eval(row["normalize"]),
            eval(row["diameter"]),
            eval(row["do_3D"]),
        )
