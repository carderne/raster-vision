import torch

from torch.nn.modules.loss import _Loss
from torch.nn import functional as F


class JaccardLoss(_Loss):
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, input, target):
        return jaccard_loss(input, target)


def jaccard_loss(input, target, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        target: a tensor of shape [B, H, W] or [B, 1, H, W].
        input: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """

    num_classes = input.shape[1]

    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[target.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(input)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[target.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(input, dim=1)

    true_1_hot = true_1_hot.type(input.type())
    dims = (0,) + tuple(range(2, target.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = 1 - (intersection / (union + eps)).mean()

    return jacc_loss
