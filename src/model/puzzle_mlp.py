import torch.nn as nn
from src.util.config import Config


class PuzzleMLP(nn.Module):
    def __init__(self, config: Config):
        super(PuzzleMLP, self).__init__()

        layers = []
        in_channels = (
            config.BOARD_REPRESENTATION.value[0]
            * config.PUZZLE_REPRESENTATION.value
        )
        height, width = (
            config.BOARD_REPRESENTATION.value[1],
            config.BOARD_REPRESENTATION.value[2],
        )

        prev_layer = in_channels * height * width
        for layer_size in config.FULLY_CONNECTED_LAYERS:
            layers.append(nn.Linear(prev_layer, layer_size))
            layers.append(config.ACTIVATION_FUNCTION())
            layers.append(nn.Dropout(config.DROPOUT))
            prev_layer = layer_size
        layers.append(nn.Linear(prev_layer, len(config.PUZZLE_CLASSES)))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
