import torch
import torch.nn as nn
from src.util.config import Config


class PuzzleCNN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        in_channels = (
            config.BOARD_REPRESENTATION.value[0]
            * config.PUZZLE_REPRESENTATION.value
        )
        height, width = (
            config.BOARD_REPRESENTATION.value[1],
            config.BOARD_REPRESENTATION.value[2],
        )

        conv_layers = []
        for out_channels, kernel, stride, padding in config.CONVOLUTION_LAYERS:
            conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )
            conv_layers.append(config.ACTIVATION_FUNCTION())
            conv_layers.append(nn.Dropout(config.DROPOUT))

            height = (
                (height + 2 * padding - kernel) // stride
                + 1
            )
            width = (
                (width + 2 * padding - kernel) // stride
                + 1
            )
            in_channels = out_channels

        self.convolutions = nn.Sequential(*conv_layers)

        flat_dim = in_channels * height * width

        fc_layers = []
        prev = flat_dim
        for hidden in config.FULLY_CONNECTED_LAYERS:
            fc_layers.append(nn.Linear(prev, hidden))
            fc_layers.append(config.ACTIVATION_FUNCTION())
            fc_layers.append(nn.Dropout(config.DROPOUT))
            prev = hidden

        fc_layers.append(nn.Linear(prev, len(config.PUZZLE_CLASSES)))
        self.fully_connected = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.convolutions(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fully_connected(x)
        return x
