"""
hic_model.py

My minimal model: a small fully-convolutional CNN, same-size in and
out (256x256 -> 256x256), single channel. No pooling/upsampling because this
is not changing the matrix's dimensions, just refining the values, so
every conv layer uses "same" padding to preserve spatial size throughout.
"""


import torch.nn as nn


class SimpleEnhanceCNN(nn.Module):
    def __init__(self, hidden_channels = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(hidden_channels, 1, kernel_size = 3, padding = 1)
        )

    def forward(self, x):
        return self.net(x)