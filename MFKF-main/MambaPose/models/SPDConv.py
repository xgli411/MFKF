import torch.nn as nn
import torch
from models.common import Conv

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc=64, outc=128, dimension=1):
        super(SPDConv, self).__init__()
        self.d = dimension
        self.conv = Conv(4*inc, outc, 3, 1, None, 1, nn.LeakyReLU(0.1))

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], self.d))


