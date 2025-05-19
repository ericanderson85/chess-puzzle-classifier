# import torch
# from torch import nn


# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1):
#         super().__init__()
#         self.smooth = smooth

#     def forward(self, logits, targets):
#         probs = torch.sigmoid(logits)
#         targets = targets.float()
#         intersection = (probs * targets).sum()
#         dice = (2. * intersection + self.smooth) / (
#             probs.sum() + targets.sum() + self.smooth
#         )
#         return 1 - dice
