import csv
import os

from prettytable import PrettyTable


class ResultHandler:
    def __init__(self, result_path):
        self.result_path = result_path

    def log_result(self, properties, are, precision, recall, f1):
        """Log a new result to the CSV file."""
        properties['are'] = are
        properties['precision'] = precision
        properties['recall'] = recall
        properties['f1'] = f1

        fieldnames = list(properties.keys())

        file_exists = os.path.exists(self.result_path)

        with open(self.result_path, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists or os.stat(self.result_path).st_size == 0:
                writer.writeheader()
            writer.writerow(properties)

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
                    f"{float(row[field]):.2f}" if field in ['are', 'precision', 'recall', 'f1'] else row[field] for
                    field in reader.fieldnames]
                table.add_row(formatted_row)
            print(table)
