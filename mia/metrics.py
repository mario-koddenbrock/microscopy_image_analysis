import numpy as np
from cellpose.metrics import aggregated_jaccard_index, average_precision


def jaccard_score(ground_truth, masks):
    aji_scores = aggregated_jaccard_index(ground_truth, masks)
    return np.mean(aji_scores)


def f1_score(ground_truth, masks):
    ap, tp, fp, fn = average_precision(ground_truth, masks)
    precision = np.sum(tp) / np.sum(tp + fp)
    recall = np.sum(tp) / np.sum(tp + fn)
    if precision + recall == 0:
        fscore = 0
    else:
        fscore = 2 * (precision * recall) / (precision + recall)
    return fscore


def simple_iou(ground_truth, masks):
    intersection = np.logical_and(ground_truth > 0, masks > 0).sum()
    union = np.logical_or(ground_truth > 0, masks > 0).sum()
    simple_jaccard = intersection / union if union > 0 else 0
    return simple_jaccard

def dice_coefficient(ground_truth, prediction):
    """Calculate Dice coefficient for a single mask."""
    intersection = np.logical_and(ground_truth > 0, prediction > 0).sum()
    total = ground_truth.sum() + prediction.sum()
    return 2 * intersection / total if total > 0 else 0
