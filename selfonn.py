"""Minimal 1-D Self-Organized Operational Neural Network (Self-ONN) layer.

A Self-ONN neuron replaces the linear convolution of a CNN neuron with a
Q-th order Maclaurin (power-series) approximation of a nodal function
(BRSR-OpGAN paper, Eqs. 3-5):

    y = b + sum_{q=1..Q} Conv1D(w_q, x^q)

which is computed here as a single convolution over the channel-wise
concatenation [x, x^2, ..., x^Q]. With Q = 1 the layer is an ordinary
Conv1d.

Parameter names and shapes (``weight``: [out, Q*in, k], ``bias``: [out])
match the layout used when the released models were trained with the
FastONN library, so the released state dicts load directly and give the
same outputs.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfONN1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=True, q=1):
        super().__init__()
        if q < 1:
            raise ValueError("q must be >= 1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.q = q
        self.weight = nn.Parameter(torch.empty(out_channels, q * in_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("tanh"))
        if self.bias is not None:
            fan_in = self.weight.shape[1] * self.weight.shape[2]
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.q > 1:
            x = torch.cat([x ** p for p in range(1, self.q + 1)], dim=1)
        return F.conv1d(x, self.weight, self.bias, stride=self.stride,
                        padding=self.padding, dilation=self.dilation)

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}, q={self.q}")
