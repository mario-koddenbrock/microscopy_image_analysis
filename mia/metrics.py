import numpy as np
from cellpose.metrics import average_precision
from medpy.metric import jc
from scipy.optimize import linear_sum_assignment


def jaccard(ground_truth, masks):
    unique_gt = np.unique(ground_truth[ground_truth > 0])
    unique_masks = np.unique(masks[masks > 0])

    if len(unique_gt) == 0 or len(unique_masks) == 0:
        return 0

    aji = np.zeros((len(unique_gt), len(unique_masks)))
    for label_gt in unique_gt:
        for label_mask in unique_masks:
            gt_mask = ground_truth == label_gt
            pred_mask = masks == label_mask
            aji[label_gt - 1, label_mask - 1] = jc(gt_mask, pred_mask)
    # Match ground truth and predicted instances using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-aji)  # Maximize Jaccard score
    # Compute mean Jaccard score for matched pairs
    matched_scores = aji[row_ind, col_ind]
    mean_score = matched_scores.mean()
    return mean_score


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
    try:

        if ground_truth is None or masks is None:
            return -2

        intersection = np.logical_and(ground_truth > 0, masks > 0).sum()
        union = np.logical_or(ground_truth > 0, masks > 0).sum()
        simple_jaccard = intersection / union if union > 0 else 0
        return simple_jaccard
    except Exception as e:
        print(f"Error: {e}")
        return -1


def dice_coefficient(ground_truth, prediction):
    """Calculate Dice coefficient for a single mask."""
    intersection = np.logical_and(ground_truth > 0, prediction > 0).sum()
    total = ground_truth.sum() + prediction.sum()
    return 2 * intersection / total if total > 0 else 0
