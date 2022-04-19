import paddle
import paddle.nn as nn
import math
from paddle.nn.utils import weight_norm


class distLinear(nn.Layer):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias_attr=False)
        self.class_wise_learnable_norm = True  # See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            weight_norm(self.L, 'weight', dim=0)  # split the weight update component to direction and norm

        if outdim <= 200:
            self.scale_factor = 2  # a fixed scale factor to scale the output of cos value into a reasonably large input for softmax,
            # for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github
        else:
            self.scale_factor = 10  # in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x_norm = paddle.norm(x, p=2, axis=1).unsqueeze(1).expand_as(x)
        x_normalized = x.divide(x_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        # matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise
        # learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor * (cos_dist)

        return scores


class Flatten(nn.Layer):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape([x.shape[0], -1])


# Simple Conv Block
class ConvBlock(nn.Layer):
    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.kernel_size = 3
        n = self.kernel_size * self.kernel_size * outdim
        weight_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Normal(mean=0.0, std=math.sqrt(2.0 / float(n))))
        self.C = nn.Conv2D(indim, outdim, self.kernel_size, padding=padding, weight_attr=weight_attr)

        weight_attr = paddle.framework.ParamAttr(
            initializer=nn.initializer.Constant(value=1.0))
        bias_attr = paddle.framework.ParamAttr(
            initializer=nn.initializer.Constant(value=0.0))
        self.BN = nn.BatchNorm2D(outdim, weight_attr=weight_attr, bias_attr=bias_attr)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2D(2)
            self.parametrized_layers.append(self.pool)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNet(nn.Layer):
    def __init__(self, depth, flatten=True):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self, x):
        out = self.trunk(x)
        return out


def Conv4():
    return ConvNet(4)
