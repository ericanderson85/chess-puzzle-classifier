import torch.nn as nn
from src.util.config import ACTIVATION_FUNCTION, CONVOLUTION_LAYERS, DROPOUT, FULLY_CONNECTED_LAYERS


class PuzzleCNN(nn.Module):
    def __init__(self):
        super(PuzzleCNN, self).__init__()

        convolutional_layers = []
        for in_channels, out_channels, kernel_size, stride, padding in CONVOLUTION_LAYERS:
            convolutional_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            convolutional_layers.append(ACTIVATION_FUNCTION())
            convolutional_layers.append(nn.Dropout(DROPOUT))
        self.convolutions = nn.Sequential(*convolutional_layers)

        fully_connected_layers = []
        for i, (in_features, out_features) in enumerate(FULLY_CONNECTED_LAYERS):
            fully_connected_layers.append(nn.Linear(in_features, out_features))
            if i < len(FULLY_CONNECTED_LAYERS) - 1:
                fully_connected_layers.append(ACTIVATION_FUNCTION())
                fully_connected_layers.append(nn.Dropout(DROPOUT))
        self.fully_connected = nn.Sequential(*fully_connected_layers)

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x
