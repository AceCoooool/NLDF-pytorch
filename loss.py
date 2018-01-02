import torch
from torch import nn
import torch.nn.functional as F


class GradLayer(nn.Module):
    def __init__(self):
        super(GradLayer, self).__init__()
        self.grad_x = nn.Conv2d(2, 2, 3, padding=1, bias=False)
        self.grad_y = nn.Conv2d(2, 2, 3, padding=1, bias=False)
        self.set_weight()

    def set_weight(self):
        x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        weight_x = nn.Parameter(x, requires_grad=False)
        weight_y = nn.Parameter(y, requires_grad=False)
        self.grad_x.weight, self.grad_y.weight = weight_x, weight_y

    def forward(self, x):
        x1, x2 = self.grad_x(x), self.grad_y(x)
        return torch.sqrt(torch.pow(x1, 2) + torch.pow(x2, 2))


class Loss_boundary(nn.Module):
    def __init__(self, contour_th):
        super(Loss, self).__init__()
        self.cth = contour_th
        self.gradlayer = GradLayer()

    def forward(self, x, label):
        prob_grad = F.tanh(self.gradlayer(x))
        label_grad = torch.gt(self.gradlayer(label), self.cth).float()
        inter = torch.sum(prob_grad * label_grad)
        union = torch.pow(prob_grad, 2).sum() + torch.pow(label_grad, 2).sum()
        # TODO: will cause inf ?
        # union = prob_grad.sum() + label_grad.sum()
        boundary_loss = (1 - 2 * (inter + 1) / (union + 1)) if inter.data[0] > 0 else 0
        sailency_loss = F.binary_cross_entropy(x, label)
        return boundary_loss, sailency_loss


class Loss_space(nn.Module):
    def __init__(self):
        super(Loss_space, self).__init__()

    def forward(self, x, label):
        inter = (x * label).sum()
        union = x.sum() + label.sum()
        dice_loss = 1 - 2 * (inter + 1) / (union + 1)
        sail_loss = F.binary_cross_entropy(x, label)
        return dice_loss, sail_loss


def build_loss(space=True):
    if space:
        return Loss_space()
    else:
        return Loss_boundary(1.5)

