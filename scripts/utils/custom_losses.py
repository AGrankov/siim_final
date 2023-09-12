import torch
from torch import nn
from torch.nn import functional as F


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.modules.BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.bce_loss(input, target) + (1 - dice_loss(input, target))

class BCELogDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.modules.BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.bce_loss(input, target) - torch.log(dice_loss(input, target))

class DiceLoss(nn.Module):
    def forward(self, input, target):
        return (1 - dice_loss(input, target))

class LogDiceLoss(nn.Module):
    def forward(self, input, target):
        return - torch.log(dice_loss(input, target))

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()

class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()
