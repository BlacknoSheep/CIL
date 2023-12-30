import torch
import torch.nn as nn


class DoubleLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
