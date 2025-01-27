import os

from mia.results import print_best_config_per_image
from mia.viz import plot_eval

main_folder = "Datasets/P013T/"

# get all the subfolder
result_path = os.path.join(main_folder, "results.csv")

if not os.path.exists(result_path):
    print("No results to display.")
    exit()


print_best_config_per_image(result_path)
plot_eval(result_path)
