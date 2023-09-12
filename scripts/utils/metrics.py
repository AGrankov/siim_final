import numpy as np

def iou(img_true, img_pred):
    i = np.sum((img_true * img_pred) > 0)
    u = np.sum((img_true + img_pred) > 0) + 0.0000000000000000001  # avoid division by zero
    return i/u

def dice(targs, pred):
    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1.0)

def correct_dice(targs, pred):
    n = pred.shape[0]
    pred = pred.reshape((n, -1))
    targs = targs.reshape((n, -1))
    intersect = (pred*targs).sum(axis=-1).astype(np.float32)
    union = (pred+targs).sum(axis=-1).astype(np.float32)
    u0 = union==0
    intersect[u0] = 1
    union[u0] = 2
    return (2. * intersect / union).mean()

def dice_torch(targs, preds):
    n = preds.shape[0]
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    intersect = (preds * targs).sum(-1).float()
    union = (preds+targs).sum(-1).float()
    u0 = union==0
    intersect[u0] = 1
    union[u0] = 2
    return (2. * intersect / union)
