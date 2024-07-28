import numpy as np

def dice_score(pred_mask, gt_mask):
    assert pred_mask.shape == gt_mask.shape
    logical_xnor = np.logical_not(np.logical_xor(pred_mask, gt_mask))
    return 2 * (logical_xnor.sum()) / (pred_mask.size + gt_mask.size)
