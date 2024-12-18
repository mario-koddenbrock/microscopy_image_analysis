import glob
import os

from mia.results import ResultHandler, print_best_config_per_image
from mia.viz import plot_aggregated_metric_variation, plot_best_scores_barplot

main_folder = "Datasets/P013T/"

# get all the subfolder
result_path = os.path.join(main_folder, "results.csv")

print_best_config_per_image(result_path, metric='f1', output_file=result_path.replace('results.csv', 'best_configs.json'))
plot_aggregated_metric_variation(result_path, metric='f1', boxplot=True)
plot_aggregated_metric_variation(result_path, metric='f1', boxplot=False)
plot_best_scores_barplot(result_path, metric='f1', output_file=result_path.replace('results.csv', 'best_scores_barplot.png'))
