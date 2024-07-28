import numpy as np

def dice_score(pred_mask, gt_mask):
    assert pred_mask.shape == gt_mask.shape
    logical_and = np.logical_and(pred_mask, gt_mask)
    logical_not_and = np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask))
    return 2 * (logical_and.sum() + logical_not_and.sum()) / (pred_mask.size + gt_mask.size)
