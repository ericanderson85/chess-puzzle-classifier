import torch.nn as nn
from src.util.config import Config


class PuzzleLogisticRegression(nn.Module):
    def __init__(self, config: Config) -> None:
        super(PuzzleLogisticRegression, self).__init__()

        in_channels = (
            config.BOARD_REPRESENTATION.value[0]
            * config.PUZZLE_REPRESENTATION.value
        )
        height, width = (
            config.BOARD_REPRESENTATION.value[1],
            config.BOARD_REPRESENTATION.value[2],
        )

        self.linear = nn.Linear(in_channels * height * width, len(config.PUZZLE_CLASSES))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
