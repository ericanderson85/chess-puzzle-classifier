from enum import Enum, auto
from chess import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
from torch import nn, optim
import os

import torch

# Directories and files paths
ENGINES_DIRECTORY = "engines/"
DATA_DIRECTORY = "data/"
GAMES_DIRECTORY = os.path.join(DATA_DIRECTORY, 'games')
PUZZLES_DIRECTORY = os.path.join(DATA_DIRECTORY, 'puzzles')
OPENING_BOOK_PATH = os.path.join(DATA_DIRECTORY, "codekiddy.bin")
ENGINE_PATHS = {
    "stockfish": os.path.join(ENGINES_DIRECTORY, "stockfish"),
    "lc0": os.path.join(ENGINES_DIRECTORY, "lc0"),
}


# Game simulation parameters
SIMULATE_GAMES_LOG_PATH = "logs/simulate_games.log"
CURRENT_ENGINES = ("stockfish", "stockfish")  # (white, black)
GAME_COUNT = 65536
USE_GAME_TIME = False
GAME_TIME_SECONDS = 2
INCREMENT_SECONDS = 0
MOVE_DEPTH = 5

# Puzzle analysis parameters
FIND_PUZZLES_LOG_PATH = "logs/find_puzzles.log"
PUZZLE_ANALYSIS_ENGINE = "stockfish"
ANALYSIS_DEPTH = 14
EVALUATION_THRESHOLD = 350
MIN_MATERIAL_GAIN = 210
MIN_WHITE_BETTER_THAN_NEXT_MOVE = 300
PUZZLE_PLY = 3

# Puzzle classification parameters
CLASSIFY_PUZZLES_LOG_PATH = "logs/classify_puzzles.log"


# Piece values for material evaluation
PAWN_VALUE = 100
KNIGHT_VALUE = 300
BISHOP_VALUE = 300
ROOK_VALUE = 500
QUEEN_VALUE = 900
KING_VALUE = 1000
PIECE_VALUES = {
    PAWN: PAWN_VALUE,
    KNIGHT: KNIGHT_VALUE,
    BISHOP: BISHOP_VALUE,
    ROOK: ROOK_VALUE,
    QUEEN: QUEEN_VALUE,
    KING: KING_VALUE
}

PIECE_INDICES_MAP = {
    'p': -1,
    'P': 1,
    'n': -2,
    'N': 2,
    'b': -3,
    'B': 3,
    'r': -4,
    'R': 4,
    'q': -5,
    'Q': 5,
    'k': -6,
    'K': 6,
}

# Concurrency settings
CPU_COUNT = os.cpu_count()


# Model parameters
PUZZLE_CLASSES = [
    'alignment',  # pins, skewers, xray, discovered
    'fork',       # one piece attacks two or more targets
    'promotion',  # pawn promotions
]

DATA_SPLIT = {
    'train': 0.8,
    'validate': 0.2,
    'test': 0
}


class LearningType(Enum):
    SUPERVISED = auto()
    SEMI_SUPERVISED = auto()


class BoardRepresentation(Enum):
    PIECE_INDEX = (1, 8, 8)
    SIGNED_ONE_HOT = (6, 8, 8)
    ONE_HOT = (12, 8, 8)


FEW_PUZZLE_COUNT = 4


class PuzzleRepresentation(Enum):
    FIRST = 1
    FIRST_AND_LAST = 2
    FEW = FEW_PUZZLE_COUNT


LEARNING_TYPE = LearningType.SUPERVISED
UNSUPERVISED_LOSS_WEIGHT = 0.5
LABEL_CONFIDENCE_THRESHOLD = 0.90

BOARD_REPRESENTATION = BoardRepresentation.PIECE_INDEX
PUZZLE_REPRESENTATION = PuzzleRepresentation.FEW

FLATTENED_CHANNEL_DIMENSION = \
    BOARD_REPRESENTATION.value[0] * PUZZLE_REPRESENTATION.value

CONVOLUTION_LAYERS = [
    # (in_channels, out_channels, kernel_size, stride, padding)
    (FLATTENED_CHANNEL_DIMENSION, 32, 3, 1, 1),
    (32,                          64, 3, 1, 1),
    (64,                         128, 3, 2, 1),

]


def conv2d_output_size(size, kernel_size, stride, padding):
    return (size + 2 * padding - kernel_size) // stride + 1


height, width = BOARD_REPRESENTATION.value[1], BOARD_REPRESENTATION.value[2]
out_channels = FLATTENED_CHANNEL_DIMENSION
for in_c, out_c, kernel, stride, pad in CONVOLUTION_LAYERS:
    height = conv2d_output_size(height, kernel, stride, pad)
    width = conv2d_output_size(width, kernel, stride, pad)
    out_channels = out_c

FLATTENED_CONVOLUTION_OUTPUT_DIMENSION = out_channels * height * width

FULLY_CONNECTED_LAYERS = [
    (FLATTENED_CONVOLUTION_OUTPUT_DIMENSION, 256),
    (256, 64),
    (64, len(PUZZLE_CLASSES)),
]

DROPOUT = 0.5
ACTIVATION_FUNCTION: nn.Module = nn.ReLU


# Training hyperparameters
ACTIVE_LEARNING_LOG_PATH = "logs/active_learning.log"
TRAIN_LOG_PATH = "logs/train.log"
RANDOM_SEED = 63
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
REGULARIZATION = 0
LOSS_FUNCTION: nn.Module = nn.CrossEntropyLoss
OPTIMIZATION_FUNCTION: optim.Optimizer = optim.Adam
